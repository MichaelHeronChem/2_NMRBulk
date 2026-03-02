import nmrglue as ng
import numpy as np
import warnings
from nmrglue.process.proc_autophase import autops
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


def process_fid(dic, data, shift=0.0):
    """
    Applies zero-filling, line-broadening, FFT, selected phasing strategy
    (manual, ACME, peak_minima, edge_symmetry, or edge_anchor), and masked polynomial auto-baseline correction.
    """
    data = np.asarray(data).ravel()

    # 1. Line Broadening
    if config.LINE_BROADENING > 0:
        sw = float(dic["procpar"]["sw"]["values"][0])
        data = ng.proc_base.em(data, lb=config.LINE_BROADENING / sw)

    # 2. Zero Filling
    zf_size = 1 << (data.size * 2).bit_length()
    data = ng.proc_base.zf_size(data, zf_size)

    # 3. Fourier Transform
    data = ng.proc_base.fft(data)

    # Calculate the referenced PPM scale for precise pivoting and masking
    ppm_scale, _, _ = get_ppm_scale(dic, data)
    ppm_scale_ref = ppm_scale + shift

    # 4. Phasing Strategy
    phase_mode = getattr(config, "PHASE_MODE", "manual")

    if phase_mode == "edge_anchor":
        valid_mask = (ppm_scale_ref < config.EXCLUDE_PPM_MIN) | (
            ppm_scale_ref > config.EXCLUDE_PPM_MAX
        )
        try:
            p0, p1, pivot_idx = edge_phasing.autophase_anchor_twist(
                data,
                ppm_scale_ref,
                valid_mask,
                anchor_ppm=getattr(config, "ANCHOR_PPM", 11.0),
                edge_threshold=getattr(config, "EDGE_THRESHOLD", 0.10),
            )
            data = ng.proc_base.ps(data, p0=p0, p1=p1)

            # Store the exact anchor ppm in the dictionary so the plotting script can find it
            if pivot_idx is not None:
                if "processing_info" not in dic:
                    dic["processing_info"] = {}
                dic["processing_info"]["anchor_ppm"] = ppm_scale_ref[pivot_idx]

        except Exception as e:
            print(
                f"  -> Auto-phase (edge_anchor) failed, falling back to unphased: {e}"
            )

    elif phase_mode == "edge_symmetry":
        valid_mask = (ppm_scale_ref < config.EXCLUDE_PPM_MIN) | (
            ppm_scale_ref > config.EXCLUDE_PPM_MAX
        )
        try:
            # Calculate pivot index for P1-only optimization
            pivot_index = None
            if hasattr(config, "PIVOT_PPM") and config.PIVOT_PPM is not None:
                pivot_index = np.argmin(np.abs(ppm_scale_ref - config.PIVOT_PPM))

            p0, p1 = edge_phasing.autophase_edge_symmetry(
                data,
                valid_mask,
                edge_threshold=getattr(config, "EDGE_THRESHOLD", 0.10),
                initial_p0=0.0,  # Start P0 at 0
                initial_p1=getattr(config, "P1", 0.0),
                pivot_index=pivot_index,
                optimize_p0=False,  # Force algorithm to only optimize P1 around the pivot
            )
            data = ng.proc_base.ps(data, p0=p0, p1=p1)
        except Exception as e:
            print(
                f"  -> Auto-phase (edge_symmetry) failed, falling back to unphased: {e}"
            )

    elif phase_mode in ["acme", "peak_minima"]:
        # Create a mask for valid regions (KEEP points OUTSIDE the exclusion zone)
        valid_mask = (ppm_scale_ref < config.EXCLUDE_PPM_MIN) | (
            ppm_scale_ref > config.EXCLUDE_PPM_MAX
        )

        # Zero out the suppressed region exclusively for phase evaluation
        data_for_phasing = data.copy()
        data_for_phasing[~valid_mask] = 0.0 + 0.0j

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, phases = autops(data_for_phasing, fn=phase_mode, return_phases=True)
            data = ng.proc_base.ps(data, p0=phases[0], p1=phases[1])
        except Exception as e:
            print(
                f"  -> Auto-phase ({phase_mode}) failed, falling back to unphased: {e}"
            )

    elif phase_mode == "manual":
        # Manual Phasing (with Pivot)
        if config.P0 != 0.0 or config.P1 != 0.0:
            if hasattr(config, "PIVOT_PPM") and config.PIVOT_PPM is not None:
                # Find the exact array index that corresponds to the user's pivot ppm
                pivot_index = np.argmin(np.abs(ppm_scale_ref - config.PIVOT_PPM))
                N = data.shape[-1]

                # Offset the starting p0 so that the phase at the pivot equals config.P0
                effective_p0 = config.P0 - config.P1 * (pivot_index / (N - 1))
            else:
                effective_p0 = config.P0

            data = ng.proc_base.ps(data, p0=effective_p0, p1=config.P1)

    # Take the real part after FT and Phasing for baselining
    data = data.real

    # 5. Auto-Baseline (3rd order polynomial fit on valid regions only)
    if getattr(config, "APPLY_BASELINE", True):
        valid_mask = (ppm_scale_ref < config.EXCLUDE_PPM_MIN) | (
            ppm_scale_ref > config.EXCLUDE_PPM_MAX
        )
        x_indices = np.arange(len(data))

        if np.any(valid_mask):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Fit the polynomial strictly to the non-solvent regions
                coeffs = np.polyfit(x_indices[valid_mask], data[valid_mask], deg=3)

            # Subtract the generated baseline curve across the ENTIRE spectrum
            baseline = np.polyval(coeffs, x_indices)
            data = data - baseline

    data /= data.size
    return dic, data


def find_ppm_shift(scout_dir):
    """
    Finds the chemical shift discrepancy using the unsupressed solvent peak.
    Calculates how much the ppm scale needs to shift so the max peak aligns with SOLVENT_PPM.
    """
    dic, data = ng.varian.read(scout_dir)
    dic, data = process_fid(dic, data)
    ppm_scale, _, _ = get_ppm_scale(dic, data)

    # Find the index of the absolute largest peak (assumed to be the unsupressed solvent)
    max_index = np.argmax(np.abs(data))

    # Return the required shift amount
    return config.SOLVENT_PPM - ppm_scale[max_index]
