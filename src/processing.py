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





def process_fid(dic, data, shift=0.0):

    """

    Applies zero-filling, FFT, edge_anchor phasing strategy,

    and masked polynomial auto-baseline correction.

    """

    data = np.asarray(data).ravel()



    # 1. Zero Filling

    zf_size = 1 << (data.size * 2).bit_length()

    data = ng.proc_base.zf_size(data, zf_size)



    # 2. Fourier Transform

    data = ng.proc_base.fft(data)



    # Calculate the referenced PPM scale for precise pivoting and masking

    ppm_scale, _, _ = get_ppm_scale(dic, data)

    ppm_scale_ref = ppm_scale + shift



    # 3. Phasing Strategy (Edge Anchor Only)

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



    # Take the real part after FT and Phasing for baselining

    data = data.real



    # 4. Auto-Baseline (3rd order polynomial fit on valid regions only)

    if getattr(config, "APPLY_BASELINE", True):

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