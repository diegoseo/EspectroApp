#!/usr/bin/env bash
set -euo pipefail

# Run this script from the EspectroApp project root.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d ".venv" ]]; then
    echo "ERROR: .venv was not found in $PROJECT_ROOT"
    echo "Create or restore the virtual environment before building."
    exit 1
fi

source .venv/bin/activate

python -m pip install --upgrade pyinstaller pillow

rm -rf build dist EspectroApp.spec

pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --onedir \
  --name EspectroApp \
  --paths src \
  --add-data "src/icom:icom" \
  --splash "packaging/linux/espectroapp_boot_splash.png" \
  --splash-center primary \
  src/app.py

echo
echo "Build completed."
echo "Run:"
echo "  ./dist/EspectroApp/EspectroApp"
