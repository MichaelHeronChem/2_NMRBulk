import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import nmrglue as ng
from scipy.signal import find_peaks  # Added for peak picking

import config
from processing import process_fid, get_ppm_scale, find_ppm_shift

# --- Paths ---
BASE_DIR = Path("data/raw")
REACTION_DIR = BASE_DIR / "reaction"
ALDEHYDE_DIR = BASE_DIR / "aldehydes"
AMINE_DIR = BASE_DIR / "amines"
PROCESSED_DIR = Path("data/processed")


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


# --- Peak Picking Logic ---

def get_region_max(ppm, data, min_ppm=5.0, max_ppm=12.0):
    """Gets the max intensity only in the downfield region, ignoring massive solvent peaks."""
    mask = (ppm >= min_ppm) & (ppm <= max_ppm)
    return np.max(data[mask]) if np.any(mask) else np.max(data)


def get_pure_aldehyde_peak(ppm, data):
    """Finds the leftmost singlet (highest ppm peak) in the pure aldehyde spectrum."""
    # Typically aldehyde protons are between 8.5 and 11.5 ppm
    mask = (ppm >= 8.5) & (ppm <= 11)
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return None
    
    sub_data = data[valid_indices]
    region_max = get_region_max(ppm, data)
    
    peaks, _ = find_peaks(sub_data, height=0.05 * region_max, prominence=0.02 * region_max)
    
    if len(peaks) > 0:
        peak_ppms = ppm[valid_indices[peaks]]
        return np.max(peak_ppms)  # Leftmost peak
    else:
        # Fallback if find_peaks misses it
        local_max_idx = np.argmax(sub_data)
        if sub_data[local_max_idx] > 0.05 * region_max:
            return ppm[valid_indices[local_max_idx]]
    return None


def check_peak_presence(target_ppm, sm_tuple, tol=0.2):
    """Checks if a peak exists in the starting material spectrum at the target ppm."""
    if sm_tuple is None:
        return False
    ppm, data, _ = sm_tuple
    mask = (ppm >= target_ppm - tol) & (ppm <= target_ppm + tol)
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return False
    
    sub_data = data[valid_indices]
    region_max = get_region_max(ppm, data)
    
    # 1. Check if there is a distinct peak in this region (matches imine detection sensitivity)
    peaks, _ = find_peaks(sub_data, height=0.005 * region_max, prominence=0.002 * region_max)
    if len(peaks) > 0:
        return True
        
    # 2. Fallback: if there's no sharp peak, check if there's significant raw intensity 
    # (Using 0.5% of the aromatic region max to block small SM humps from becoming imines)
    local_max = np.max(sub_data)
    if local_max > 0.005 * region_max:
        return True
        
    return False


def find_reaction_peaks(rxn_ppm, rxn_data, pure_ald_tuple, pure_amine_tuple):
    """Finds the aldehyde and imine peaks in the reaction spectrum."""
    results = {"aldehyde": None, "imine": None}
    
    pure_ald_ppm = None
    if pure_ald_tuple:
        p_ppm, p_data, _ = pure_ald_tuple
        pure_ald_ppm = get_pure_aldehyde_peak(p_ppm, p_data)
        
    # 1. Identify Reaction Aldehyde Peak
    rxn_ald_ppm = None
    if pure_ald_ppm is not None:
        # Should show up in reaction spectra +/- 0.2 ppm
        mask = (rxn_ppm >= pure_ald_ppm - 0.2) & (rxn_ppm <= pure_ald_ppm + 0.2)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) > 0:
            local_max_idx = valid_indices[np.argmax(rxn_data[valid_indices])]
            rxn_ald_ppm = rxn_ppm[local_max_idx]
            results["aldehyde"] = rxn_ald_ppm
            
    # 2. Identify Reaction Imine Peak
    if rxn_ald_ppm is not None:
        search_max = rxn_ald_ppm - 0.05  # Must be strictly to the right of aldehyde
        search_min = 7.0  # Broadened slightly to safely capture imine
        
        mask = (rxn_ppm >= search_min) & (rxn_ppm <= search_max)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            sub_data = rxn_data[valid_indices]
            region_max = get_region_max(rxn_ppm, rxn_data)
            
            # Use sensitivity relative to the aromatic region, not the entire spectrum
            peaks, _ = find_peaks(sub_data, height=0.005 * region_max, prominence=0.002 * region_max)
            
            if len(peaks) > 0:
                peak_ppms = rxn_ppm[valid_indices[peaks]]
                
                # Sort by distance to the aldehyde peak (closest first)
                distances = np.abs(rxn_ald_ppm - peak_ppms)
                sorted_idx = np.argsort(distances)
                peak_ppms = peak_ppms[sorted_idx]
                
                for p_ppm in peak_ppms:
                    # Check if peak is unique to reaction (not in amine or aldehyde)
                    in_amine = check_peak_presence(p_ppm, pure_amine_tuple)
                    in_ald = check_peak_presence(p_ppm, pure_ald_tuple)
                    
                    if not in_amine and not in_ald:
                        results["imine"] = p_ppm
                        break  # Found the closest unique singlet to the right
                    
    return results


def main():
    if not REACTION_DIR.exists():
        print(f"Directory not found: {REACTION_DIR}. Ensure data/raw structure exists.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- MEMORY CACHES ---
    # Store processed pure spectra here so we only compute them once!
    processed_amines = {}
    processed_aldehydes = {}

    for reaction_path in REACTION_DIR.iterdir():
        if not reaction_path.is_dir():
            continue

        folder_name = reaction_path.name
        match = re.match(r"(\d+)_am_(\d+)_ald", folder_name)
        if not match:
            continue

        amine_num, ald_num = match.groups()
        presat_dir = reaction_path / "PRESAT_01.fid"
        scout_dir = reaction_path / "scoutfids" / "PRESAT_01_Scout1D.fid"

        if not presat_dir.exists() or not scout_dir.exists():
            print(f"Skipping {folder_name}: Missing PRESAT_01.fid or scoutfids/PRESAT_01_Scout1D.fid.")
            continue

        print(f"\nProcessing Imine Reaction: {folder_name}")

        # 1. Process Reaction Spectrum
        shift = find_ppm_shift(str(scout_dir))
        rxn_ppm, rxn_data, rxn_anchor = get_spectrum(presat_dir, shift)
        spectra_to_plot = {"Reaction": (rxn_ppm, rxn_data, rxn_anchor)}

        # 2. Process Pure Components (WITH CACHING)
        amine_path = AMINE_DIR / f"{amine_num}_amine"
        ald_path = ALDEHYDE_DIR / f"{ald_num}_aldehyde"

        if amine_path.exists():
            if str(amine_path) not in processed_amines:
                try:
                    # Process and store in cache
                    processed_amines[str(amine_path)] = resolve_pure_spectrum(amine_path, is_amine=True)
                except Exception as e:
                    print(f"  Warning: Could not process amine {amine_path}: {e}")
            
            # Pull from cache
            if str(amine_path) in processed_amines:
                spectra_to_plot["Pure Amine"] = processed_amines[str(amine_path)]

        if ald_path.exists():
            if str(ald_path) not in processed_aldehydes:
                try:
                    # Process and store in cache
                    processed_aldehydes[str(ald_path)] = resolve_pure_spectrum(ald_path, is_amine=False)
                except Exception as e:
                    print(f"  Warning: Could not process aldehyde {ald_path}: {e}")
            
            # Pull from cache
            if str(ald_path) in processed_aldehydes:
                spectra_to_plot["Pure Aldehyde"] = processed_aldehydes[str(ald_path)]

        # 3. Plotting Setup
        fig, ax = plt.subplots(figsize=(10, 6))

        if getattr(config, "AUTO_X_LIMITS", True):
            all_ppms = np.concatenate([ppm for ppm, _, _ in spectra_to_plot.values()])
            plot_x_max = np.max(all_ppms)
            plot_x_min = np.min(all_ppms)
        else:
            plot_x_max = getattr(config, "FIXED_X_MAX", 12.0)
            plot_x_min = getattr(config, "FIXED_X_MIN", -1.0)

        # --- Normalize spectra ---
        if getattr(config, "NORMALIZE_SPECTRA", True):
            for label in list(spectra_to_plot.keys()):
                ppm, data, anchor = spectra_to_plot[label]
                mask = (ppm >= plot_x_min) & (ppm <= plot_x_max)
                visible_max = np.max(np.abs(data[mask])) if np.any(mask) else np.max(np.abs(data))
                
                if visible_max > 0:
                    # Creating a new tuple with the normalized data safely avoids mutating the cache
                    spectra_to_plot[label] = (ppm, data / visible_max, anchor)

        # --- Identify Specific Peaks (Aldehyde & Imine) ---
        rxn_tuple = spectra_to_plot.get("Reaction")
        ald_tuple = spectra_to_plot.get("Pure Aldehyde")
        amine_tuple = spectra_to_plot.get("Pure Amine")
        
        peak_labels = {"aldehyde": None, "imine": None}
        if rxn_tuple:
            peak_labels = find_reaction_peaks(rxn_tuple[0], rxn_tuple[1], ald_tuple, amine_tuple)

        # 4. Plotting (Stacked)
        if getattr(config, "NORMALIZE_SPECTRA", True):
            offset_step = 1.1
        else:
            rxn_ppm, rxn_data, _ = spectra_to_plot["Reaction"]
            rxn_mask = (rxn_ppm >= plot_x_min) & (rxn_ppm <= plot_x_max)
            offset_step = np.max(rxn_data[rxn_mask]) * 1.1 if np.any(rxn_mask) else np.max(rxn_data) * 1.1

        y_offset = 0
        max_y_plotted, min_y_plotted = -np.inf, np.inf

        for label, (ppm, data, anchor) in spectra_to_plot.items():
            plotted_data = data + y_offset
            
            mask = (ppm >= plot_x_min) & (ppm <= plot_x_max)
            if np.any(mask):
                visible_max, visible_min = np.max(plotted_data[mask]), np.min(plotted_data[mask])
            else:
                visible_max, visible_min = np.max(plotted_data), np.min(plotted_data)
                
            max_y_plotted = max(max_y_plotted, visible_max)
            min_y_plotted = min(min_y_plotted, visible_min)

            ax.axhline(y=y_offset, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.plot(ppm, plotted_data, label=label, linewidth=0.3, color="black")
            
            # --- Draw Annotations ---
            if label == "Reaction":
                # Label Aldehyde peak
                if peak_labels["aldehyde"]:
                    ald_ppm = peak_labels["aldehyde"]
                    idx = np.abs(ppm - ald_ppm).argmin()
                    peak_y = plotted_data[idx]
                    if plot_x_min <= ald_ppm <= plot_x_max:
                        ax.plot(ppm[idx], peak_y, marker='*', color='blue', markersize=8)
                        ax.text(ppm[idx], peak_y + (offset_step * 0.08), f"{ald_ppm:.2f}", 
                                color='blue', fontsize=9, ha='center', va='bottom', zorder=5)
                
                # Label Imine peak
                if peak_labels["imine"]:
                    imine_ppm = peak_labels["imine"]
                    idx = np.abs(ppm - imine_ppm).argmin()
                    peak_y = plotted_data[idx]
                    if plot_x_min <= imine_ppm <= plot_x_max:
                        ax.plot(ppm[idx], peak_y, marker='^', color='green', markersize=7)
                        ax.text(ppm[idx], peak_y + (offset_step * 0.08), f"{imine_ppm:.2f}", 
                                color='green', fontsize=9, ha='center', va='bottom', zorder=5)

            y_offset -= offset_step

        ax.set_xlim(plot_x_max, plot_x_min)
        if max_y_plotted != -np.inf and min_y_plotted != np.inf:
            ax.set_ylim(min_y_plotted - 0.1 * offset_step, max_y_plotted + 0.1 * offset_step)

        ax.set_xlabel("Chemical Shift (ppm)", fontsize=12)
        ax.set_ylabel("Normalized Intensity" if getattr(config, "NORMALIZE_SPECTRA", True) else "Intensity (a.u.)", fontsize=12)
        ax.set_title(f"Imine Formation: {folder_name}", fontsize=14)

        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])

        plt.tight_layout()
        out_file = PROCESSED_DIR / f"plot_{folder_name}.png"
        plt.savefig(out_file, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"  --> Saved plot: {out_file}")


if __name__ == "__main__":
    main()