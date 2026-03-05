import nmrglue as ng
import numpy as np
import warnings
import config
import edge_phasing


def get_ppm_scale(dic, data):
    """Calculates the PPM axis."""
    try:
        sf = float(dic["procpar"]["sreffrq"]["values"][0])
        sw = float(dic["procpar"]["sw"]["values"][0])
    except (KeyError, IndexError):
        sf, sw = 400.0, 4000.0
    size = data.shape[-1]
    freq_hz = np.linspace(sw / 2, -sw / 2, size)
    return freq_hz / sf, sf, sw


def process_fid(dic, data, shift=0.0, anchor_target=None, fast_mode=False):
    """
    Applies zero-filling, FFT, edge_anchor phasing, and auto-baseline.
    If fast_mode=True, it skips phasing and baselining, returning magnitude data (for speed).
    """
    data = np.asarray(data).ravel()

    # 1. Zero Filling
    zf_size = 1 << (data.size * 2).bit_length()
    data = ng.proc_base.zf_size(data, zf_size)

    # 2. Fourier Transform
    data = ng.proc_base.fft(data)

    ppm_scale, _, _ = get_ppm_scale(dic, data)
    ppm_scale_ref = ppm_scale + shift

    # --- FAST MODE SHORTCIRCUIT ---
    if fast_mode:
        data = np.abs(data)  # Magnitude is fine for finding the peak center
        data /= data.size
        return dic, data

    # 3. Phasing Strategy (Edge Anchor Only)
    valid_mask = (ppm_scale_ref < config.EXCLUDE_PPM_MIN) | (
        ppm_scale_ref > config.EXCLUDE_PPM_MAX
    )
   
    if anchor_target is None:
        anchor_target = getattr(config, "ANCHOR_PPM", 10.0)

    try:
        p0, p1, pivot_idx = edge_phasing.autophase_anchor_twist(
            data,
            ppm_scale_ref,
            valid_mask,
            anchor_ppm=anchor_target,
            edge_threshold=getattr(config, "EDGE_THRESHOLD", 0.10),
        )
        # We also need to supply the shift array to apply_phase here
        shift_array = np.arange(len(data)) / (len(data) - 1)
        data = edge_phasing.apply_phase(data, p0, p1, shift_array)

        if pivot_idx is not None:
            if "processing_info" not in dic:
                dic["processing_info"] = {}
            dic["processing_info"]["anchor_ppm"] = ppm_scale_ref[pivot_idx]

    except Exception as e:
        print(f"  -> Auto-phase (edge_anchor) failed, falling back to unphased: {e}")

    data = data.real

    # 4. Auto-Baseline
    if getattr(config, "APPLY_BASELINE", True):
        x_indices = np.arange(len(data))
        if np.any(valid_mask):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coeffs = np.polyfit(x_indices[valid_mask], data[valid_mask], deg=3)
            baseline = np.polyval(coeffs, x_indices)
            data = data - baseline

    data /= data.size
    return dic, data


def find_ppm_shift(scout_dir):
    """
    Finds the chemical shift discrepancy using the unsupressed solvent peak.
    Uses fast_mode=True to skip expensive phasing math.
    """
    dic, data = ng.varian.read(scout_dir)
    # Engage Fast Mode!
    dic, data = process_fid(dic, data, fast_mode=True)
    ppm_scale, _, _ = get_ppm_scale(dic, data)

    max_index = np.argmax(np.abs(data))
    return config.SOLVENT_PPM - ppm_scale[max_index]