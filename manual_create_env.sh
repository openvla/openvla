#!/usr/bin/env bash
#
# Creates a Python 3.10 virtual environment, installs dependencies,
# and downloads the OpenVLA model. Intended for environments
# without Docker support where GPU is Slurm-managed (e.g. entropy).
# Designed to be used in conjunction with vla-scripts/finetune.sub.
#
# Usage: ./manual_create_env.sh
#
# Assumptions:
# - This script is run from the root directory of the openvla_finetuner project.
# - The user has a working Python 3.10 and `virtualenv` installation.
# - The user has internet access to download dependencies and the model.
# - The user has Slurm installed and configured.

set -e
set -o pipefail

function check_virtualenv {
  if ! command -v virtualenv &> /dev/null; then
    echo "Error: virtualenv is not installed" >&2
    exit 1
  fi
}

function verify_directory {
  if [[ ! -f "$(basename "$0")" ]]; then
    echo "Error: This script must be run from the openvla/ directory" >&2
    exit 1
  fi
}

function setup_virtualenv {
  if [[ ! -d ".venv" ]]; then
    virtualenv -p 3.10 .venv || exit 1
  fi
  source .venv/bin/activate || exit 1
  pip install --upgrade pip || exit 1
  pip install "setuptools<60" || exit 1  # Or else build dlimp_openvla will fail.
}

function install_dependencies {
  # Note: We intentionally don't exit on pip check error below due to known
  # dependency conflicts between OpenVLA (which needs torch==2.2.0) and LeRobot
  # (which needs torch>=2.2.1). These conflicts are expected and the
  # installation will still work for our purposes.

  # Install OpenVLA.
  pip install -e . || exit 1

  # Install LeRobot.
  pushd third_party/lerobot || exit 1
  pip install -e . || exit 1
  popd || exit 1

  # Reinstall OpenVLA dependencies that LeRobot may have overwritten.
  set +e  # Temporarily disable exit on error.
  pip check | awk '$1 ~ /openvla/ {gsub(/,/,"",$5); print $5}' | \
    xargs pip install
  set -e  # Re-enable exit on error.

  # Install Flash Attention.
  pip install packaging ninja || exit 1
  # We will install flash-attn inside the Slurm job, after loading CUDA.
  # This avoids errors when CUDA_HOME is not set during environment creation.
  # pip install \
  #   "flash-attn==2.5.5" \
  #   --no-build-isolation || exit 1
}

function download_model {
  pip install huggingface-hub || exit 1
  huggingface-cli download openvla/openvla-7b || exit 1
}

function main {
  check_virtualenv
  verify_directory
  setup_virtualenv
  install_dependencies
  download_model
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"