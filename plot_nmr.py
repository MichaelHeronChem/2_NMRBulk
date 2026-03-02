import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import nmrglue as ng

import config
from processing import process_fid, get_ppm_scale, find_ppm_shift

# --- Paths ---
BASE_DIR = Path("data/raw")
REACTION_DIR = BASE_DIR / "reaction"
ALDEHYDE_DIR = BASE_DIR / "aldehyde"
AMINE_DIR = BASE_DIR / "amine"
PROCESSED_DIR = Path("data/processed")


def get_spectrum(dir_path, shift=0.0):
    """Helper to load, process, reference (via shift parameter), and optionally normalize a spectrum."""
    dic, data = ng.varian.read(str(dir_path))

    # The processing script now dynamically handles auto vs manual routing internally
    dic, data = process_fid(dic, data, shift=shift)

    ppm_scale, _, _ = get_ppm_scale(dic, data)
    ppm_scale = ppm_scale + shift

    if config.NORMALIZE_SPECTRA:
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

    return ppm_scale, data


def resolve_pure_spectrum(base_path):
    """
    Attempts to load a pure spectrum.
    1. Looks for the unsupressed scoutfid to calculate the reference shift.
    2. Applies that exact shift to the PRESAT_01.fid spectrum.
    """
    scout_dir = base_path / "scoutfids" / "PRESAT_01_Scout1D.fid"
    presat_dir = base_path / "PRESAT_01.fid"

    if scout_dir.exists() and presat_dir.exists():
        # Find the shift required to set the largest scoutfid peak to Acetonitrile
        shift = find_ppm_shift(str(scout_dir))

        # Shift the PRESAT fid spectrum by this exact amount
        return get_spectrum(presat_dir, shift)
    else:
        # Fallback if the folder structure is just a standard fid without a scout
        return get_spectrum(base_path, shift=0.0)


def main():
    if not REACTION_DIR.exists():
        print(f"Directory not found: {REACTION_DIR}. Ensure data/raw structure exists.")
        return

    # Ensure the output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Iterate through all reaction folders
    for reaction_path in REACTION_DIR.iterdir():
        if not reaction_path.is_dir():
            continue

        folder_name = reaction_path.name

        # Parse the folder name (e.g., "1_am_2_ald")
        match = re.match(r"(\d+)_am_(\d+)_ald", folder_name)
        if not match:
            continue

        amine_num, ald_num = match.groups()
        presat_dir = reaction_path / "PRESAT_01.fid"
        scout_dir = reaction_path / "scoutfids" / "PRESAT_01_Scout1D.fid"

        if not presat_dir.exists() or not scout_dir.exists():
            print(
                f"Skipping {folder_name}: Missing PRESAT_01.fid or scoutfids/PRESAT_01_Scout1D.fid."
            )
            continue

        print(f"\nProcessing Imine Reaction: {folder_name}")

        # 1. Process Reaction Spectrum
        # Calculate the referencing shift using the unsupressed scoutfid
        shift = find_ppm_shift(str(scout_dir))

        # Apply this exact shift to the corresponding PRESAT spectrum
        rxn_ppm, rxn_data = get_spectrum(presat_dir, shift)

        spectra_to_plot = {"Reaction": (rxn_ppm, rxn_data)}

        # 2. Process Pure Components
        amine_path = AMINE_DIR / f"{amine_num}_amine"
        ald_path = ALDEHYDE_DIR / f"{ald_num}_aldehyde"

        if amine_path.exists():
            try:
                spectra_to_plot["Pure Amine"] = resolve_pure_spectrum(amine_path)
            except Exception as e:
                print(f"  Warning: Could not process amine {amine_path}: {e}")

        if ald_path.exists():
            try:
                spectra_to_plot["Pure Aldehyde"] = resolve_pure_spectrum(ald_path)
            except Exception as e:
                print(f"  Warning: Could not process aldehyde {ald_path}: {e}")

        # 3. Plotting (Stacked)
        fig, ax = plt.subplots(figsize=(10, 6))

        y_offset = 0
        offset_step = 1.1 if config.NORMALIZE_SPECTRA else np.max(rxn_data) * 1.1
        max_y_plotted = -np.inf

        for label, (ppm, data) in spectra_to_plot.items():
            plotted_data = data + y_offset
            # Track the highest peak for y-axis limits
            max_y_plotted = max(max_y_plotted, np.max(plotted_data))

            # Add a horizontal line at the 0-intensity baseline for this specific spectrum
            ax.axhline(
                y=y_offset, color="gray", linestyle="--", linewidth=0.5, alpha=0.7
            )

            # Use black color and extremely thin lines for fine baseline detail
            ax.plot(ppm, plotted_data, label=label, linewidth=0.3, color="black")
            y_offset += offset_step

        # Standard NMR Convention (ppm goes right to left)
        ax.invert_xaxis()

        # Restrict x-axis dynamically or use fixed ranges
        if getattr(config, "AUTO_X_LIMITS", True):
            # Find the absolute min and max ppm across all data we just plotted
            all_ppms = np.concatenate([ppm for ppm, _ in spectra_to_plot.values()])
            ax.set_xlim(np.max(all_ppms), -0.5)
        else:
            ax.set_xlim(
                getattr(config, "FIXED_X_MAX", 12.0),
                getattr(config, "FIXED_X_MIN", -1.0),
            )

        # Restrict y-axis dynamically based on the tallest peak in the whole plot
        if max_y_plotted != -np.inf:
            ax.set_ylim(-0.2 * max_y_plotted, 1.1 * max_y_plotted)

        ax.set_xlabel("Chemical Shift (ppm)", fontsize=12)
        ax.set_ylabel(
            "Normalized Intensity" if config.NORMALIZE_SPECTRA else "Intensity (a.u.)",
            fontsize=12,
        )
        ax.set_title(f"Imine Formation: {folder_name}", fontsize=14)

        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])  # Hide Y ticks as it's a stack plot

        plt.tight_layout()

        # Save to the new processed directory with high DPI
        out_file = PROCESSED_DIR / f"plot_{folder_name}.png"
        plt.savefig(out_file, dpi=600)
        plt.close()
        print(f"  --> Saved plot: {out_file}")


if __name__ == "__main__":
    main()
