import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nmrglue as ng

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# Processing Toggles
APPLY_PHASING = False
APPLY_BASELINE = False
APPLY_SUPPRESSION = False
PLOT_IN_PPM = True  # Changed to True for standard NMR visualization

# Visual Parameters
LINE_WIDTH = 0.5
PNG_DPI = 300
Y_UPPER_BUFFER = 1.2  # 1.2x the max peak
Y_LOWER_BUFFER = -1.0  # 10% below baseline for visual clarity

# Target ppm for the Acetonitrile solvent peak
SOLVENT_PPM = 1.98

# Processing Parameters
LINE_BROADENING = 0.5

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def process_fid(dic, data):
    data = np.asarray(data).ravel()
    if LINE_BROADENING > 0:
        sw = float(dic["procpar"]["sw"]["values"][0])
        # nmrglue em uses Hz/sw for lb
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
    # Solvent peak is usually the largest in un-suppressed scout scans
    max_index = np.argmax(np.abs(data))
    return SOLVENT_PPM - ppm_scale[max_index]


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    sample_folders = glob.glob(os.path.join(RAW_DATA_DIR, "*_am_*_ald"))

    for folder in sample_folders:
        sample_name = os.path.basename(folder)
        print(f"\nProcessing Sample: {sample_name}")

        presat_dir = os.path.join(folder, "PRESAT_01.fid")
        scout_dir = os.path.join(folder, "scoutfids", "PRESAT_01_Scout1D.fid")

        if not os.path.exists(presat_dir) or not os.path.exists(scout_dir):
            print(f"  -> Skipping: Missing .fid directories")
            continue

        try:
            # 1. Determine shift from Scout
            ppm_shift = find_ppm_shift(scout_dir)

            # 2. Process Presat Data
            dic, data = ng.varian.read(presat_dir)
            dic, data = process_fid(dic, data)
            ppm_scale, _, _ = get_ppm_scale(dic, data)
            ppm_scale += ppm_shift

            # Extract real part for plotting
            final_data = np.real(data)

            # --- DYNAMIC Y-AXIS CALCULATION ---
            # Find the highest peak in the visible real spectrum
            max_val = np.max(final_data)
            dynamic_ymax = max_val * Y_UPPER_BUFFER
            dynamic_ymin = max_val * Y_LOWER_BUFFER

            # --- PLOTTING LOGIC ---
            fig, ax = plt.subplots(figsize=(12, 6))

            color = "black" if PLOT_IN_PPM else "blue"
            if PLOT_IN_PPM:
                ax.plot(ppm_scale, final_data, color=color, linewidth=LINE_WIDTH)
                ax.set_xlim(14.0, -0.5)  # Standard 1H NMR range
                ax.set_xlabel("Chemical Shift (ppm)", fontweight="bold")
            else:
                ax.plot(final_data, color=color, linewidth=LINE_WIDTH)
                ax.set_xlabel("Data Points (Array Index)", fontweight="bold")

            # --- APPLY DYNAMIC LIMITS ---
            ax.set_ylim(dynamic_ymin, dynamic_ymax)
            ax.set_ylabel("Intensity (a.u.)", fontweight="bold")

            # Formatter handles scientific notation for high intensities
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

            ax.set_title(f"PRESAT Spectrum: {sample_name} (Auto-Scaled Y)", fontsize=14)
            plt.tight_layout()

            # Save SVG and PNG
            out_base = os.path.join(PROCESSED_DATA_DIR, f"{sample_name}_processed")
            plt.savefig(f"{out_base}.svg", format="svg")
            plt.savefig(f"{out_base}.png", dpi=PNG_DPI)

            plt.close(fig)
            print(f"  -> Exported: Peak Max {max_val:.2e} | Y-Limit {dynamic_ymax:.2e}")

        except Exception as e:
            print(f"  -> Error processing {sample_name}: {e}")


if __name__ == "__main__":
    main()
