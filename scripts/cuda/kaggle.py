# Kaggle kernel for the CUDA run (M6). scripts/cuda/kaggle.sh fills in COMMIT and pushes this
# file; it clones that commit, runs scripts/cuda/run.sh, and leaves results in /kaggle/working.
import subprocess
import sys

REPO = "https://github.com/akshajnad/rust_llm_inference_server.git"
COMMIT = "@COMMIT@"
WORK = "/tmp/rlis"  # builds and models stay out of /kaggle/working, which is saved as output

script = f"""set -euo pipefail
git clone --quiet {REPO} {WORK}
cd {WORK}
git checkout --quiet {COMMIT}
OUT=/kaggle/working/cuda scripts/cuda/run.sh
"""
sys.exit(subprocess.run(["bash", "-c", script]).returncode)
