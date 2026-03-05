import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import nmrglue as ng
from scipy.signal import find_peaks

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


# --- Peak Picking Logic ---

def get_region_max(ppm, data, min_ppm=5.0, max_ppm=12.0):
    """Gets the max intensity only in the downfield region, ignoring massive solvent peaks."""
    mask = (ppm >= min_ppm) & (ppm <= max_ppm)
    return np.max(data[mask]) if np.any(mask) else np.max(data)


def get_pure_aldehyde_peak(ppm, data):
    """
    Finds the aldehyde peak in the pure spectrum using a tiered search:
    1. Look for peaks in 9.5 to 10.5 ppm.
    2. If none, look in 10.5 to 11.5 ppm.
    3. If none, look below 9.5 ppm (down to 8.5 ppm).
    Inside the successful tier, the peak closest to 10.0 ppm is selected.
    """
    region_max = get_region_max(ppm, data)
    
    # Define the tiers in order of priority
    tiers = [
        (9.5, 10.5),
        (10.5, 11.5),
        (8.5, 9.5)
    ]
    
    for min_p, max_p in tiers:
        mask = (ppm >= min_p) & (ppm <= max_p)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            continue
            
        sub_data = data[valid_indices]
        # Find peaks meeting significance criteria
        peaks, _ = find_peaks(sub_data, height=0.05 * region_max, prominence=0.02 * region_max)
        
        if len(peaks) > 0:
            peak_ppms = ppm[valid_indices[peaks]]
            # Within this tier, pick the peak closest to 10.0 ppm
            idx_closest = np.argmin(np.abs(peak_ppms - 10.0))
            return peak_ppms[idx_closest]
            
    # Final global fallback if tiers fail: highest point in the entire 8.5-11.5 region
    mask_all = (ppm >= 8.5) & (ppm <= 11.5)
    valid_indices = np.where(mask_all)[0]
    if len(valid_indices) > 0:
        local_max_idx = valid_indices[np.argmax(data[valid_indices])]
        if data[local_max_idx] > 0.05 * region_max:
            return ppm[local_max_idx]
            
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
    
    peaks, _ = find_peaks(sub_data, height=0.005 * region_max, prominence=0.002 * region_max)
    if len(peaks) > 0:
        return True
        
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
        
    rxn_ald_ppm = None
    if pure_ald_ppm is not None:
        mask = (rxn_ppm >= pure_ald_ppm - 0.2) & (rxn_ppm <= pure_ald_ppm + 0.2)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) > 0:
            local_max_idx = valid_indices[np.argmax(rxn_data[valid_indices])]
            rxn_ald_ppm = rxn_ppm[local_max_idx]
            results["aldehyde"] = rxn_ald_ppm
            
    if rxn_ald_ppm is not None:
        search_max = rxn_ald_ppm - 0.05
        search_min = 7.0
        
        mask = (rxn_ppm >= search_min) & (rxn_ppm <= search_max)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            sub_data = rxn_data[valid_indices]
            region_max = get_region_max(rxn_ppm, rxn_data)
            peaks, _ = find_peaks(sub_data, height=0.005 * region_max, prominence=0.002 * region_max)
            
            if len(peaks) > 0:
                peak_ppms = rxn_ppm[valid_indices[peaks]]
                distances = np.abs(rxn_ald_ppm - peak_ppms)
                sorted_idx = np.argsort(distances)
                peak_ppms = peak_ppms[sorted_idx]
                
                for p_ppm in peak_ppms:
                    in_amine = check_peak_presence(p_ppm, pure_amine_tuple)
                    in_ald = check_peak_presence(p_ppm, pure_ald_tuple)
                    
                    if not in_amine and not in_ald:
                        results["imine"] = p_ppm
                        break
                    
    return results


def process_block(block_path, processed_amines_cache, processed_aldehydes_cache):
    """Processes all reactions within a single block folder found in reaction/."""
    block_name = block_path.name
    
    processed_dir = PROCESSED_BASE_DIR / block_name
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Entering {block_name}")

    for reaction_path in block_path.iterdir():
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
            print(f"Skipping {folder_name}: Missing FIDs.")
            continue

        print(f"  Processing Imine Reaction: {folder_name}")

        shift = find_ppm_shift(str(scout_dir))
        rxn_ppm, rxn_data, rxn_anchor = get_spectrum(presat_dir, shift)
        spectra_to_plot = {"Reaction": (rxn_ppm, rxn_data, rxn_anchor)}

        # Reference the global amines and aldehydes folders
        amine_path = AMINE_DIR / f"{amine_num}_amine"
        ald_path = ALDEHYDE_DIR / f"{ald_num}_aldehyde"

        if amine_path.exists():
            if str(amine_path) not in processed_amines_cache:
                try:
                    processed_amines_cache[str(amine_path)] = resolve_pure_spectrum(amine_path, is_amine=True)
                except Exception as e:
                    print(f"    Warning: Amine {amine_num} failed: {e}")
            if str(amine_path) in processed_amines_cache:
                spectra_to_plot["Pure Amine"] = processed_amines_cache[str(amine_path)]

        if ald_path.exists():
            if str(ald_path) not in processed_aldehydes_cache:
                try:
                    processed_aldehydes_cache[str(ald_path)] = resolve_pure_spectrum(ald_path, is_amine=False)
                except Exception as e:
                    print(f"    Warning: Aldehyde {ald_num} failed: {e}")
            if str(ald_path) in processed_aldehydes_cache:
                spectra_to_plot["Pure Aldehyde"] = processed_aldehydes_cache[str(ald_path)]

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))

        if getattr(config, "AUTO_X_LIMITS", True):
            all_ppms = np.concatenate([ppm for ppm, _, _ in spectra_to_plot.values()])
            plot_x_max, plot_x_min = np.max(all_ppms), np.min(all_ppms)
        else:
            plot_x_max, plot_x_min = getattr(config, "FIXED_X_MAX", 12.0), getattr(config, "FIXED_X_MIN", -1.0)

        if getattr(config, "NORMALIZE_SPECTRA", True):
            for label in list(spectra_to_plot.keys()):
                ppm, data, anchor = spectra_to_plot[label]
                mask = (ppm >= plot_x_min) & (ppm <= plot_x_max)
                v_max = np.max(np.abs(data[mask])) if np.any(mask) else np.max(np.abs(data))
                if v_max > 0:
                    spectra_to_plot[label] = (ppm, data / v_max, anchor)

        rxn_tuple = spectra_to_plot.get("Reaction")
        peak_labels = find_reaction_peaks(rxn_tuple[0], rxn_tuple[1], spectra_to_plot.get("Pure Aldehyde"), spectra_to_plot.get("Pure Amine"))

        offset_step = 1.1 if getattr(config, "NORMALIZE_SPECTRA", True) else np.max(rxn_tuple[1]) * 1.1
        y_offset = 0
        max_y, min_y = -np.inf, np.inf

        for label, (ppm, data, anchor) in spectra_to_plot.items():
            plotted_data = data + y_offset
            mask = (ppm >= plot_x_min) & (ppm <= plot_x_max)
            vis_max, vis_min = (np.max(plotted_data[mask]), np.min(plotted_data[mask])) if np.any(mask) else (np.max(plotted_data), np.min(plotted_data))
            max_y, min_y = max(max_y, vis_max), min(min_y, vis_min)

            ax.axhline(y=y_offset, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.plot(ppm, plotted_data, label=label, linewidth=0.3, color="black")
            
            if label == "Reaction":
                if peak_labels["aldehyde"]:
                    a_ppm = peak_labels["aldehyde"]
                    idx = np.abs(ppm - a_ppm).argmin()
                    ax.plot(ppm[idx], plotted_data[idx], marker='*', color='blue', markersize=8)
                    ax.text(ppm[idx], plotted_data[idx] + (offset_step * 0.08), f"{a_ppm:.2f}", color='blue', fontsize=9, ha='center')
                if peak_labels["imine"]:
                    i_ppm = peak_labels["imine"]
                    idx = np.abs(ppm - i_ppm).argmin()
                    ax.plot(ppm[idx], plotted_data[idx], marker='^', color='green', markersize=7)
                    ax.text(ppm[idx], plotted_data[idx] + (offset_step * 0.08), f"{i_ppm:.2f}", color='green', fontsize=9, ha='center')

            y_offset -= offset_step

        ax.set_xlim(plot_x_max, plot_x_min)
        ax.set_ylim(min_y - 0.1 * offset_step, max_y + 0.1 * offset_step)
        ax.set_xlabel("Chemical Shift (ppm)")
        ax.set_ylabel("Normalized Intensity" if getattr(config, "NORMALIZE_SPECTRA", True) else "Intensity (a.u.)")
        ax.set_title(f"Imine Formation: {folder_name} ({block_name})")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])

        plt.tight_layout()
        out_file = processed_dir / f"plot_{folder_name}.png"
        plt.savefig(out_file, dpi=600, bbox_inches="tight")
        plt.close()


def main():
    if not REACTION_BASE_DIR.exists():
        print(f"Reaction directory {REACTION_BASE_DIR} not found.")
        return

    # Look for folders starting with 'block_' inside 'data/raw/reaction/'
    block_folders = [d for d in REACTION_BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("Block_")]
    
    if not block_folders:
        print("No block directories found in data/raw/reaction/.")
        return

    # Cache starting materials across all blocks for efficiency
    processed_amines_cache = {}
    processed_aldehydes_cache = {}

    for block_folder in block_folders:
        process_block(block_folder, processed_amines_cache, processed_aldehydes_cache)


if __name__ == "__main__":
    main()