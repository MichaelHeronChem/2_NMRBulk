# ==========================================
# NMR Processing Parameters
# ==========================================

# --- General ---
LINE_BROADENING = 0
SOLVENT_PPM = 1.96  # Acetonitrile (CD3CN) standard chemical shift

# --- Phasing Mode ---
# Select the phasing strategy:
# 'edge_symmetry' -> Auto-phasing by minimizing asymmetry of all peak edges
# 'edge_anchor'   -> Optimizes P0 on a specific anchor peak, then optimizes P1 globally
PHASE_MODE = "edge_anchor"

# --- Edge Anchor / Symmetry Parameters ---
ANCHOR_PPM = 10  # The target ppm for the anchor peak (used in 'edge_anchor')
EDGE_THRESHOLD = 0.10  # Fraction of peak max to define edges

# --- Auto-Phasing & Auto-Baseline Exclusion Zone ---
# Ignores the massive solvent/water peaks between these values.
EXCLUDE_PPM_MIN = 0.5
EXCLUDE_PPM_MAX = 4.5

# --- Baseline Correction ---
# Uses masked polynomial auto-baseline
APPLY_BASELINE = True

# --- Plotting Preferences ---
# Normalizing to the maximum peak makes visual comparison
# between Amine, Aldehyde, and Reaction much easier.
NORMALIZE_SPECTRA = True

# --- Plot X-Axis Limits ---
# Set to True to automatically fit the x-axis to the exact bounds of the acquired data.
# Set to False to enforce the fixed standard limits below (e.g., 12 to 4 ppm).
AUTO_X_LIMITS = False
FIXED_X_MAX = 12.0
FIXED_X_MIN = 4.0