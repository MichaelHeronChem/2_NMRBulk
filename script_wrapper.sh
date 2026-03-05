#!/bin/bash
#SBATCH --job-name=nmr_plot                 # Job name
#SBATCH --output=nmr_plot_%j.out            # Standard output log (%j = job ID)
#SBATCH --error=nmr_plot_%j.err             # Standard error log (%j = job ID)
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks (usually 1 for a single python script)
#SBATCH --cpus-per-task=1                   # Increased CPU count for multi-threaded math libraries
#SBATCH --mem=16G                           # Generous memory allocation
#SBATCH --time=00:30:00                     # Shorter time = much faster queue scheduling on Hamilton
# #SBATCH --partition=shared           # Omitted: Let Hamilton use the default compute partition

# ==========================================================
# Environment Setup
# ==========================================================
# Ensure 'uv' is available in your PATH. 
# Depending on your HPC, you might need to load a module:
# module load python uv  # Check Hamilton docs for exact module names

# Alternatively, if you installed uv locally to your user directory:
# export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# ==========================================================
# Optimize NumPy/SciPy Multi-threading
# ==========================================================
# Force underlying math libraries (FFT, polyfit) to use all requested CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ==========================================================
# Job Execution
# ==========================================================
echo "Starting NMR processing job on $(hostname) at $(date)"
echo "Working directory: $(pwd)"

# Run the Python script using uv's managed environment
uv run python src/plot_nmr.py

echo "Job finished at $(date)"