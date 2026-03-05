import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import nmrglue as ng
import pandas as pd
from scipy.signal import find_peaks, peak_widths

import config
from processing import process_fid, get_ppm_scale, find_ppm_shift

# --- Paths ---
BASE_DIR = Path("data/raw")
REACTION_BASE_DIR = BASE_DIR / "reaction"
ALDEHYDE_DIR = BASE_DIR / "aldehydes"
AMINE_DIR = BASE_DIR / "amines"
PROCESSED_BASE_DIR = Path("data/processed")


def get_spectrum(dir_path, shift=0.0, anchor_target=None):
    """Helper to load, process, and reference (via shift parameter) a spectrum."""
    dic, data = ng.varian.read(str(dir_path))
    dic, data = process_fid(dic, data, shift=shift, anchor_target=anchor_target)
    ppm_scale, _, _ = get_ppm_scale(dic, data)
    ppm_scale = ppm_scale + shift
    anchor_used = dic.get("processing_info", {}).get("anchor_ppm", None)
    return ppm_scale, data, anchor_used


def resolve_pure_spectrum(base_path, is_amine=False):
    """Attempts to load a pure spectrum."""
    scout_dir = base_path / "scoutfids" / "PRESAT_01_Scout1D.fid"
    presat_dir = base_path / "PRESAT_01.fid"
    target = "leftmost" if is_amine else None

    if scout_dir.exists() and presat_dir.exists():
        shift = find_ppm_shift(str(scout_dir))
        return get_spectrum(presat_dir, shift, anchor_target=target)
    else:
        return get_spectrum(base_path, shift=0.0, anchor_target=target)


# --- Peak Picking & Integration Logic ---

def get_region_max(ppm, data, min_ppm=5.0, max_ppm=12.0):
    """Gets the max intensity only in the downfield region, ignoring massive solvent peaks."""
    mask = (ppm >= min_ppm) & (ppm <= max_ppm)
    return np.max(data[mask]) if np.any(mask) else np.max(data)


def calculate_fwhm_ppm(ppm, data, target_ppm):
    """Calculates the Full Width at Half Maximum in ppm units for a specific peak."""
    if target_ppm is None:
        return 0.02 # Default fallback
    
    # Narrow window around target to find width safely
    window = 0.15
    mask = (ppm >= target_ppm - window) & (ppm <= target_ppm + window)
    indices = np.where(mask)[0]
    if len(indices) < 5:
        return 0.02
        
    sub_data = data[indices]
    # find_peaks relative to local sub_data
    peaks, _ = find_peaks(sub_data, height=np.max(sub_data)*0.5)
    if len(peaks) == 0:
        return 0.02
        
    # Pick the peak closest to our target in index space
    target_idx_in_sub = np.argmin(np.abs(ppm[indices[peaks]] - target_ppm))
    results_half = peak_widths(sub_data, [peaks[target_idx_in_sub]], rel_height=0.5)
    
    # results_half[0] is width in samples
    width_samples = results_half[0][0]
    ppm_per_sample = np.abs(ppm[1] - ppm[0])
    return width_samples * ppm_per_sample


def get_integration_bounds(ppm, data, target_ppm):
    """Calculates integration bounds based on 5 * FWHM."""
    if target_ppm is None:
        return None, None
    
    fwhm = calculate_fwhm_ppm(ppm, data, target_ppm)
    half_width = (5.0 * fwhm) / 2.0
    
    # Cap width to prevent integrating into neighboring peaks (max 0.3 ppm total)
    half_width = min(half_width, 0.15)
    
    return target_ppm - half_width, target_ppm + half_width


def integrate_peak_absolute(ppm, data, target_ppm):
    """
    Integrates the absolute area around a peak using 5*FWHM limits.
    """
    low, high = get_integration_bounds(ppm, data, target_ppm)
    if low is None:
        return 0.0, None, None
    
    mask = (ppm >= low) & (ppm <= high)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return 0.0, low, high
    
    # Use absolute values to account for phasing issues
    abs_data = np.abs(data[indices])
    area = np.sum(abs_data)
    return area, low, high


def get_pure_aldehyde_peak(ppm, data):
    """Finds the aldehyde peak closest to 10.0 ppm using tiered search."""
    region_max = get_region_max(ppm, data)
    tiers = [(9.5, 10.5), (10.5, 11.5), (8.5, 9.5)]
    
    for min_p, max_p in tiers:
        mask = (ppm >= min_p) & (ppm <= max_p)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0: continue
            
        sub_data = data[valid_indices]
        peaks, _ = find_peaks(sub_data, height=0.01 * region_max, prominence=0.01 * region_max)
        
        if len(peaks) > 0:
            peak_ppms = ppm[valid_indices[peaks]]
            idx_closest = np.argmin(np.abs(peak_ppms - 10.0))
            return peak_ppms[idx_closest]
            
    mask_all = (ppm >= 8.5) & (ppm <= 11.5)
    valid_indices = np.where(mask_all)[0]
    if len(valid_indices) > 0:
        local_max_idx = valid_indices[np.argmax(data[valid_indices])]
        if data[local_max_idx] > 0.01 * region_max:
            return ppm[local_max_idx]
    return None


def check_peak_presence(target_ppm, sm_tuple, tol=0.2):
    """Checks if a peak exists in starting material at target ppm."""
    if sm_tuple is None: return False
    ppm, data, _ = sm_tuple
    mask = (ppm >= target_ppm - tol) & (ppm <= target_ppm + tol)
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0: return False
    
    sub_data = data[valid_indices]
    region_max = get_region_max(ppm, data)
    peaks, _ = find_peaks(sub_data, height=0.01 * region_max, prominence=0.002 * region_max)
    if len(peaks) > 0: return True
    if np.max(sub_data) > 0.01 * region_max: return True
    return False


def find_reaction_peaks(rxn_ppm, rxn_data, pure_ald_tuple, pure_amine_tuple):
    """Finds the aldehyde and imine peaks in the reaction spectrum."""
    results = {"aldehyde": None, "imine": None}
    pure_ald_ppm = None
    if pure_ald_tuple:
        pure_ald_ppm = get_pure_aldehyde_peak(pure_ald_tuple[0], pure_ald_tuple[1])
        
    rxn_ald_ppm = None
    if pure_ald_ppm is not None:
        # Should show up in reaction spectra +/- 0.2 ppm
        mask = (rxn_ppm >= pure_ald_ppm - 0.2) & (rxn_ppm <= pure_ald_ppm + 0.2)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) > 0:
            local_max_idx = valid_indices[np.argmax(rxn_data[valid_indices])]
            rxn_ald_ppm = rxn_ppm[local_max_idx]
            results["aldehyde"] = rxn_ald_ppm
            
    if rxn_ald_ppm is not None:
        search_max, search_min = rxn_ald_ppm - 0.05, 7.0
        mask = (rxn_ppm >= search_min) & (rxn_ppm <= search_max)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            region_max = get_region_max(rxn_ppm, rxn_data)
            peaks, _ = find_peaks(rxn_data[valid_indices], height=0.01 * region_max, prominence=0.002 * region_max)
            
            if len(peaks) > 0:
                peak_ppms = rxn_ppm[valid_indices[peaks]]
                distances = np.abs(rxn_ald_ppm - peak_ppms)
                sorted_idx = np.argsort(distances)
                peak_ppms = peak_ppms[sorted_idx]
                
                for p_ppm in peak_ppms:
                    if not check_peak_presence(p_ppm, pure_amine_tuple) and not check_peak_presence(p_ppm, pure_ald_tuple):
                        results["imine"] = p_ppm
                        break
    return results


def process_block(block_path, processed_amines_cache, processed_aldehydes_cache):
    """Processes all reactions within a single block folder found in reaction/."""
    block_name = block_path.name
    processed_dir = PROCESSED_BASE_DIR / block_name
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==========================================")
    print(f"PROCESSING BLOCK: {block_name}")
    print(f"==========================================")
    
    # Get list of valid reaction folders first for progress counting
    valid_folders = []
    for d in block_path.iterdir():
        if d.is_dir() and re.match(r"(\d+)_am_(\d+)_ald", d.name):
            valid_folders.append(d)
    
    total_reactions = len(valid_folders)
    print(f"Found {total_reactions} reactions in this block.")

    results_data = []

    for i, reaction_path in enumerate(valid_folders, 1):
        folder_name = reaction_path.name
        match = re.match(r"(\d+)_am_(\d+)_ald", folder_name)
        amine_num, ald_num = match.groups()
        
        print(f"[{i}/{total_reactions}] Reaction: {folder_name}...", end=" ", flush=True)

        presat_dir, scout_dir = reaction_path / "PRESAT_01.fid", reaction_path / "scoutfids" / "PRESAT_01_Scout1D.fid"

        if not presat_dir.exists() or not scout_dir.exists():
            print("FAILED (Missing FID files)")
            continue

        try:
            shift = find_ppm_shift(str(scout_dir))
            rxn_ppm, rxn_data, rxn_anchor = get_spectrum(presat_dir, shift)
            spectra_to_plot = {"Reaction": (rxn_ppm, rxn_data, rxn_anchor)}

            for id_val, path_base, cache, is_am in [(amine_num, AMINE_DIR, processed_amines_cache, True), 
                                                    (ald_num, ALDEHYDE_DIR, processed_aldehydes_cache, False)]:
                p = path_base / (f"{id_val}_amine" if is_am else f"{id_val}_aldehyde")
                if p.exists():
                    if str(p) not in cache:
                        try: cache[str(p)] = resolve_pure_spectrum(p, is_amine=is_am)
                        except: pass
                    if str(p) in cache: spectra_to_plot["Pure Amine" if is_am else "Pure Aldehyde"] = cache[str(p)]

            # --- Peak Identification & Integration ---
            rxn_tuple = spectra_to_plot.get("Reaction")
            peak_labels = find_reaction_peaks(rxn_tuple[0], rxn_tuple[1], spectra_to_plot.get("Pure Aldehyde"), spectra_to_plot.get("Pure Amine"))

            ald_area, ald_low, ald_high = integrate_peak_absolute(rxn_tuple[0], rxn_tuple[1], peak_labels["aldehyde"])
            imine_area, imine_low, imine_high = integrate_peak_absolute(rxn_tuple[0], rxn_tuple[1], peak_labels["imine"])
            
            total_area = imine_area + ald_area
            ratio = imine_area / total_area if total_area > 0 else 0.0

            # Log peak status
            peaks_found = []
            if peak_labels["aldehyde"]: peaks_found.append("Ald")
            if peak_labels["imine"]: peaks_found.append("Imine")
            status_str = f"Found: ({', '.join(peaks_found) if peaks_found else 'None'}) Ratio: {ratio:.3f}"

            results_data.append({
                "Folder": folder_name, "Amine_ID": amine_num, "Aldehyde_ID": ald_num,
                "Aldehyde_PPM": peak_labels["aldehyde"] or 0, "Imine_PPM": peak_labels["imine"] or 0,
                "Aldehyde_Area": ald_area, "Imine_Area": imine_area, "Ratio": ratio
            })

            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_x_max, plot_x_min = (np.max(rxn_ppm), np.min(rxn_ppm)) if getattr(config, "AUTO_X_LIMITS", True) else (getattr(config, "FIXED_X_MAX", 12.0), getattr(config, "FIXED_X_MIN", -1.0))

            # Normalize for visualization
            normalized_spectra = {}
            if getattr(config, "NORMALIZE_SPECTRA", True):
                for label, (p, d, a) in spectra_to_plot.items():
                    m = (p >= plot_x_min) & (p <= plot_x_max)
                    v_max = np.max(np.abs(d[m])) if np.any(m) else np.max(np.abs(d))
                    normalized_spectra[label] = (p, d / v_max if v_max > 0 else d, a)
            else:
                normalized_spectra = spectra_to_plot

            offset_step = 1.1
            y_offset = 0
            
            # Variables to find tightest vertical limits
            min_y_seen = 0
            max_y_seen = 0

            for i, (label, (p, d, a)) in enumerate(normalized_spectra.items()):
                plotted_data = d + y_offset
                ax.axhline(y=y_offset, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
                ax.plot(p, plotted_data, label=label, linewidth=0.3, color="black")
                
                # Filter data in window for vertical bounding
                mask = (p >= plot_x_min) & (p <= plot_x_max)
                visible_data = plotted_data[mask] if np.any(mask) else plotted_data
                
                max_y_seen = max(max_y_seen, np.max(visible_data))
                min_y_seen = min(min_y_seen, np.min(visible_data))

                if label == "Reaction":
                    for ptype, p_ppm, p_low, p_high, color, marker in [
                        ("ald", peak_labels["aldehyde"], ald_low, ald_high, 'blue', '*'),
                        ("imine", peak_labels["imine"], imine_low, imine_high, 'green', '^')
                    ]:
                        if p_ppm:
                            idx = np.abs(p - p_ppm).argmin()
                            # Symbol position
                            marker_y = plotted_data[idx] + (offset_step * 0.05)
                            ax.plot(p[idx], marker_y, marker=marker, color=color, markersize=8)
                            
                            # Label position
                            label_y = marker_y + (offset_step * 0.10)
                            ax.text(p[idx], label_y, f"{p_ppm:.2f}", color=color, fontsize=9, ha='center', va='bottom')
                            
                            # Update max seen to account for the label height
                            max_y_seen = max(max_y_seen, label_y + (offset_step * 0.1))
                            
                            if p_low and p_high:
                                fill_mask = (p >= p_low) & (p <= p_high)
                                ax.fill_between(p, y_offset, plotted_data, where=fill_mask, color=color, alpha=0.15)
                    
                y_offset -= offset_step

            ax.set_xlim(plot_x_max, plot_x_min)
            
            # Apply tight vertical limits with a small 5% buffer
            y_range = max_y_seen - min_y_seen
            ax.set_ylim(min_y_seen - 0.05 * y_range, max_y_seen + 0.05 * y_range)
            
            ax.set_xlabel("Chemical Shift (ppm)")
            ax.set_ylabel("Normalized Intensity")
            ax.set_title(f"Imine Formation: {folder_name} ({block_name})")
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
            ax.grid(True, alpha=0.2)
            ax.set_yticks([])

            plt.tight_layout()
            plt.savefig(processed_dir / f"plot_{folder_name}.png", dpi=600, bbox_inches="tight")
            plt.close()
            print(f"DONE. {status_str}")
            
        except Exception as e:
            print(f"FAILED (Error: {e})")

    if results_data:
        csv_path = processed_dir / f"results_{block_name}.csv"
        pd.DataFrame(results_data).to_csv(csv_path, index=False)
        print(f"\n--> Successfully saved {len(results_data)} results to {csv_path}")


def main():
    print("Starting NMR Processing Script...")
    if not REACTION_BASE_DIR.exists():
        print(f"ERROR: Reaction directory {REACTION_BASE_DIR} not found.")
        return
        
    block_folders = sorted([d for d in REACTION_BASE_DIR.iterdir() if d.is_dir() and d.name.lower().startswith("block_")])
    if not block_folders:
        print("ERROR: No block directories found in data/raw/reaction/.")
        return

    print(f"Discovered {len(block_folders)} blocks to process: {[b.name for b in block_folders]}")

    processed_amines_cache, processed_aldehydes_cache = {}, {}
    for block_folder in block_folders:
        process_block(block_folder, processed_amines_cache, processed_aldehydes_cache)
        
    print("\n==========================================")
    print("ALL BLOCKS COMPLETED SUCCESSFULLY")
    print("==========================================")

if __name__ == "__main__":
    main()