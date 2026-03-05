import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks


def apply_phase(data, p0, p1, shift_array):
    """Applies zero and first order phase correction in degrees using a pre-calculated shift array."""
    phase_angle = np.deg2rad(p0 + p1 * shift_array)
    return data * np.exp(1j * phase_angle)


def edge_symmetry_objective(phases, data, peak_data, shift_array):
    """
    Objective function: Calculates the penalty score for given P0 and P1.
    Minimizes the difference between left and right edge intensities.
    """
    p0, p1 = phases
    phased_data = apply_phase(data, p0, p1, shift_array).real

    asymmetry_score = 0.0
    negativity_score = 0.0

    for p, l, r in peak_data:
        I_left = phased_data[l]
        I_right = phased_data[r]
        asymmetry_score += abs(I_left - I_right)

        if phased_data[p] < 0:
            negativity_score += abs(phased_data[p]) * 100

        if I_left > phased_data[p] or I_right > phased_data[p]:
            max_edge = max(I_left, I_right)
            negativity_score += abs(max_edge - phased_data[p]) * 50

    return asymmetry_score + negativity_score


def autophase_anchor_twist(
    data, ppm_scale, valid_mask, anchor_ppm=11.0, edge_threshold=0.10
):
    """
    Step 1: Finds the peak closest to the anchor_ppm and optimizes ONLY P0 for that single peak.
    Step 2: Locks that peak as the pivot point, and globally optimizes P1 for the rest of the spectrum.
    If anchor_ppm="leftmost", targets the most downfield peak instead.
    """
    magnitude = np.abs(data)
    masked_magnitude = magnitude.copy()
    masked_magnitude[~valid_mask] = 0.0

    global_max = np.max(masked_magnitude)
    if global_max == 0:
        return 0.0, 0.0, None

    peaks, _ = find_peaks(masked_magnitude, height=global_max * 0.05, distance=20)

    peak_data = []
    for p in peaks:
        target_intensity = magnitude[p] * edge_threshold
        l = p
        while l > 0 and magnitude[l] > target_intensity:
            l -= 1
        r = p
        while r < len(magnitude) - 1 and magnitude[r] > target_intensity:
            r += 1
        if l > 0 and r < len(magnitude) - 1 and (r - l) > 2:
            peak_data.append((p, l, r))

    if not peak_data:
        return 0.0, 0.0, None

    if anchor_ppm == "leftmost":
        closest_peak = max(peak_data, key=lambda p_info: ppm_scale[p_info[0]])
    else:
        closest_peak = None
        min_dist = float("inf")
        for p_info in peak_data:
            p_idx = p_info[0]
            dist = abs(ppm_scale[p_idx] - float(anchor_ppm))
            if dist < min_dist:
                min_dist = dist
                closest_peak = p_info

    if closest_peak is None:
        return 0.0, 0.0, None

    pivot_idx = closest_peak[0]
    N = len(data)
    
    # --- PRE-CALCULATE SHIFT ARRAY HERE ---
    # This prevents np.arange from running thousands of times in the loop
    shift_array = np.arange(N) / (N - 1)

    def p0_objective(p0_array):
        return edge_symmetry_objective([p0_array[0], 0.0], data, [closest_peak], shift_array)

    res_p0 = minimize(p0_objective, [0.0], method="Nelder-Mead")
    opt_p0_at_anchor = res_p0.x[0]

    def p1_objective(p1_array):
        p1 = p1_array[0]
        effective_p0 = opt_p0_at_anchor - p1 * (pivot_idx / (N - 1))
        return edge_symmetry_objective([effective_p0, p1], data, peak_data, shift_array)

    res_p1 = minimize(p1_objective, [0.0], method="Nelder-Mead")
    opt_p1 = res_p1.x[0]

    final_p0 = opt_p0_at_anchor - opt_p1 * (pivot_idx / (N - 1))

    return final_p0, opt_p1, pivot_idx