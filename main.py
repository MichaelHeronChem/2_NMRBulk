import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nmrglue as ng
import re

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
PNG_DIR = os.path.join(PROCESSED_DATA_DIR, "png")
SVG_DIR = os.path.join(PROCESSED_DATA_DIR, "svg")

DICT_DIR = "dictionaries"
AMINE_DICT_PATH = os.path.join(DICT_DIR, "amine_dictionary.tsv")
ALDEHYDE_DICT_PATH = os.path.join(DICT_DIR, "aldehyde_dictionary.tsv")

LINE_WIDTH = 0.5
PNG_DPI = 300
Y_UPPER_BUFFER = 1.35  # Room for peak labels
Y_LOWER_BUFFER = -0.1
SOLVENT_PPM = 1.98
LINE_BROADENING = 0.5

# Peak Picking Settings
PEAK_THRESHOLD_FACTOR = 0.05  # 5% of global maximum intensity
LABEL_FONT_SIZE = 7

# =============================================================================
# DICTIONARY LOADING & CLEANING
# =============================================================================


def sanitize_filename(name):
    """Removes characters that are illegal or problematic in filenames."""
    name = (
        str(name).replace(",", "-").replace("[", "").replace("]", "").replace(" ", "_")
    )
    return re.sub(r"(?u)[^-\w.]", "", name)


def load_dicts():
    amine_map = {}
    ald_map = {}
    try:
        if os.path.exists(AMINE_DICT_PATH):
            df = pd.read_csv(AMINE_DICT_PATH, sep=r"\s+", engine="python")
            amine_map = {
                str(k).strip(): str(v).strip() for k, v in zip(df["Number"], df["Name"])
            }
        if os.path.exists(ALDEHYDE_DICT_PATH):
            df = pd.read_csv(ALDEHYDE_DICT_PATH, sep=r"\s+", engine="python")
            ald_map = {
                str(k).strip(): str(v).strip() for k, v in zip(df["Number"], df["Name"])
            }
    except Exception as e:
        print(f"Error loading dictionaries: {e}")
    return amine_map, ald_map


# =============================================================================
# NMR PROCESSING FUNCTIONS
# =============================================================================


def process_fid(dic, data):
    data = np.asarray(data).ravel()
    if LINE_BROADENING > 0:
        sw = float(dic["procpar"]["sw"]["values"][0])
        data = ng.process.proc_base.em(data, lb=LINE_BROADENING / sw)
    zf_size = 1 << (data.size * 2).bit_length()
    data = ng.process.proc_base.zf_size(data, zf_size)
    data = ng.process.proc_base.fft(data)
    data /= data.size
    return dic, data


def get_ppm_scale(dic, data):
    try:
        sf = float(dic["procpar"]["sreffrq"]["values"][0])
        sw = float(dic["procpar"]["sw"]["values"][0])
    except (KeyError, IndexError):
        sf, sw = 400.0, 4000.0
    size = data.shape[-1]
    freq_hz = np.linspace(sw / 2, -sw / 2, size)
    return freq_hz / sf, sf, sw


def find_ppm_shift(scout_dir):
    dic, data = ng.varian.read(scout_dir)
    dic, data = process_fid(dic, data)
    ppm_scale, _, _ = get_ppm_scale(dic, data)
    max_index = np.argmax(np.abs(data))
    return SOLVENT_PPM - ppm_scale[max_index]


# =============================================================================
# MAIN
# =============================================================================


def main():
    os.makedirs(PNG_DIR, exist_ok=True)
    os.makedirs(SVG_DIR, exist_ok=True)

    amine_lookup, ald_lookup = load_dicts()
    sample_folders = glob.glob(os.path.join(RAW_DATA_DIR, "*_am_*_ald"))

    for folder in sample_folders:
        folder_name = os.path.basename(folder)

        try:
            parts = folder_name.split("_")
            amine_id = parts[0].strip()
            ald_id = parts[2].strip()

            amine_raw = amine_lookup.get(amine_id, f"Amine-{amine_id}")
            ald_raw = ald_lookup.get(ald_id, f"Aldehyde-{ald_id}")

            amine_clean = sanitize_filename(amine_raw)
            ald_clean = sanitize_filename(ald_raw)
            clean_filename = f"{amine_clean}_{ald_clean}"
        except Exception:
            clean_filename = folder_name
            amine_raw, ald_raw = "Unknown", "Unknown"

        print(f"Processing: {folder_name} -> {clean_filename}")

        presat_dir = os.path.join(folder, "PRESAT_01.fid")
        scout_dir = os.path.join(folder, "scoutfids", "PRESAT_01_Scout1D.fid")

        if not (os.path.exists(presat_dir) and os.path.exists(scout_dir)):
            print(f"  -> Skipping: Missing fid files.")
            continue

        try:
            # Data Processing
            ppm_shift = find_ppm_shift(scout_dir)
            dic, data = ng.varian.read(presat_dir)
            dic, data = process_fid(dic, data)
            ppm_scale, _, _ = get_ppm_scale(dic, data)
            ppm_scale += ppm_shift
            final_data = np.real(data)

            # --- Peak Picking ---
            # Use 'pthres' for the positive threshold in modern nmrglue
            thres_value = np.max(final_data) * PEAK_THRESHOLD_FACTOR
            peaks = ng.analysis.peakpick.pick(final_data, pthres=thres_value)

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(ppm_scale, final_data, color="black", linewidth=LINE_WIDTH)

            # Filter and annotate peaks
            for p in peaks:
                idx = int(p[0])
                p_ppm = ppm_scale[idx]
                p_amp = final_data[idx]

                # Only label peaks in requested regions
                if p_ppm > 6.0 or p_ppm < 1.0:
                    ax.annotate(
                        f"{p_ppm:.2f}",
                        xy=(p_ppm, p_amp),
                        xytext=(0, 8),  # 8 points vertical offset
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=LABEL_FONT_SIZE,
                        color="blue",
                        fontweight="bold",
                    )

            # Formatting
            ax.set_xlim(14.0, -0.5)
            # Adjusting Y-limit based on peak height to ensure labels fit
            ax.set_ylim(
                np.max(final_data) * Y_LOWER_BUFFER, np.max(final_data) * Y_UPPER_BUFFER
            )

            ax.set_xlabel("Chemical Shift (ppm)", fontweight="bold")
            ax.set_ylabel("Intensity (a.u.)", fontweight="bold")
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))
            ax.get_yaxis().set_visible(False)

            plt.title(f"{amine_raw} + {ald_raw}", fontsize=10)
            plt.tight_layout()

            plt.savefig(os.path.join(SVG_DIR, f"{clean_filename}.svg"), format="svg")
            plt.savefig(os.path.join(PNG_DIR, f"{clean_filename}.png"), dpi=PNG_DPI)
            plt.close(fig)

        except Exception as e:
            print(f"  -> Error: {e}")


if __name__ == "__main__":
    main()
