import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks


def apply_phase(data, p0, p1):
    """Applies zero and first order phase correction in degrees (matching nmrglue)."""
    N = len(data)
    shift = np.arange(N) / (N - 1)
    phase_angle = np.deg2rad(p0 + p1 * shift)
    return data * np.exp(1j * phase_angle)


def edge_symmetry_objective(phases, data, peak_data):
    """
    Objective function: Calculates the penalty score for given P0 and P1.
    Minimizes the difference between left and right edge intensities.
    """
    p0, p1 = phases
    phased_data = apply_phase(data, p0, p1).real

    asymmetry_score = 0.0
    negativity_score = 0.0

    for p, l, r in peak_data:
        # Absolute difference between left and right edge intensities
        I_left = phased_data[l]
        I_right = phased_data[r]
        asymmetry_score += abs(I_left - I_right)

        # Penalize heavily if the peak apex is negative (inverted peak)
        if phased_data[p] < 0:
            negativity_score += abs(phased_data[p]) * 100

        # Penalize if edges are higher than the peak (dispersion shape)
        if I_left > phased_data[p] or I_right > phased_data[p]:
            max_edge = max(I_left, I_right)
            negativity_score += abs(max_edge - phased_data[p]) * 50

    return asymmetry_score + negativity_score


def edge_symmetry_objective_p1(p1_array, data, peak_data, pivot_index):
    """
    1D Objective function: Only varies P1.
    Forces effective P0 to be exactly 0 at the defined pivot point.
    """
    p1 = p1_array[0]
    N = len(data)

    # If the user wants P0=0 at the pivot_index,
    # the mathematical P0 applied globally to the whole array must be offset.
    effective_p0 = 0.0 - p1 * (pivot_index / (N - 1))

    return edge_symmetry_objective([effective_p0, p1], data, peak_data)


def autophase_edge_symmetry(
    data,
    valid_mask,
    edge_threshold=0.10,
    initial_p0=0.0,
    initial_p1=0.0,
    pivot_index=None,
    optimize_p0=True,
):
    """
    Optimizes P0 and P1 (or just P1) by minimizing peak edge asymmetry on non-solvent peaks.
    """
    magnitude = np.abs(data)
    masked_magnitude = magnitude.copy()
    masked_magnitude[~valid_mask] = 0.0

    # 1. Detect peaks on the magnitude spectrum (location doesn't shift with phase)
    global_max = np.max(masked_magnitude)
    if global_max == 0:
        return initial_p0, initial_p1

    # Find prominent peaks (at least 5% of max)
    peaks, _ = find_peaks(masked_magnitude, height=global_max * 0.05, distance=20)

    # 2. Map the edges for each peak
    peak_data = []
    for p in peaks:
        peak_max = magnitude[p]
        target_intensity = peak_max * edge_threshold

        # Trace left
        l = p
        while l > 0 and magnitude[l] > target_intensity:
            l -= 1
        # Trace right
        r = p
        while r < len(magnitude) - 1 and magnitude[r] > target_intensity:
            r += 1

        # Keep if valid edges found
        if l > 0 and r < len(magnitude) - 1 and (r - l) > 2:
            peak_data.append((p, l, r))

    if not peak_data:
        print("  -> Edge Symmetry warning: No valid peaks found for optimization.")
        if not optimize_p0 and pivot_index is not None:
            N = len(data)
            effective_p0 = 0.0 - initial_p1 * (pivot_index / (N - 1))
            return effective_p0, initial_p1
        return initial_p0, initial_p1

    # 3. Optimize P0 and P1 (or just P1)
    if optimize_p0 or pivot_index is None:
        initial_guess = [initial_p0, initial_p1]
        result = minimize(
            edge_symmetry_objective,
            initial_guess,
            args=(data, peak_data),
            method="Nelder-Mead",
            options={"maxiter": 1000},
        )
        return result.x[0], result.x[1]
    else:
        # Optimize ONLY P1 around the user's pivot
        initial_guess = [initial_p1]
        result = minimize(
            edge_symmetry_objective_p1,
            initial_guess,
            args=(data, peak_data, pivot_index),
            method="Nelder-Mead",
            options={"maxiter": 1000},
        )
        opt_p1 = result.x[0]
        N = len(data)
        opt_p0 = 0.0 - opt_p1 * (pivot_index / (N - 1))
        return opt_p0, opt_p1


def autophase_anchor_twist(
    data, ppm_scale, valid_mask, anchor_ppm=11.0, edge_threshold=0.10
):
    """
    Step 1: Finds the peak closest to the anchor_ppm and optimizes ONLY P0 for that single peak.
    Step 2: Locks that peak as the pivot point, and globally optimizes P1 for the rest of the spectrum.
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

    # Find the peak closest to the anchor_ppm
    closest_peak = None
    min_dist = float("inf")
    for p_info in peak_data:
        p_idx = p_info[0]
        dist = abs(ppm_scale[p_idx] - anchor_ppm)
        if dist < min_dist:
            min_dist = dist
            closest_peak = p_info

    if closest_peak is None:
        return 0.0, 0.0, None

    pivot_idx = closest_peak[0]
    N = len(data)

    # Step 1: Optimize P0 exclusively for the anchor peak (P1 forced to 0)
    def p0_objective(p0_array):
        return edge_symmetry_objective([p0_array[0], 0.0], data, [closest_peak])

    res_p0 = minimize(p0_objective, [0.0], method="Nelder-Mead")
    opt_p0_at_anchor = res_p0.x[0]

    # Step 2: Optimize P1 for ALL peaks, locking the anchor peak's phase
    def p1_objective(p1_array):
        p1 = p1_array[0]
        # Calculate the required mathematical global P0 to maintain opt_p0_at_anchor at the pivot
        effective_p0 = opt_p0_at_anchor - p1 * (pivot_idx / (N - 1))
        return edge_symmetry_objective([effective_p0, p1], data, peak_data)

    res_p1 = minimize(p1_objective, [0.0], method="Nelder-Mead")
    opt_p1 = res_p1.x[0]

    # Convert back to global mathematical P0/P1 required by nmrglue
    final_p0 = opt_p0_at_anchor - opt_p1 * (pivot_idx / (N - 1))

    return final_p0, opt_p1, pivot_idx
