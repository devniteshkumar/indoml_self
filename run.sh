#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
	python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

python scripts/prepare_data.py