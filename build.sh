#!/usr/bin/env bash
# build.sh — builds RoadSight.app using PyInstaller
# Usage: bash build.sh
set -e

echo "==> Activating virtual environment..."
source .venv/bin/activate

echo "==> Installing PyInstaller..."
pip install pyinstaller --quiet

echo "==> Building RoadSight.app..."
pyinstaller RoadSight.spec --clean --noconfirm

echo ""
echo "Build complete!"
echo "  App bundle : dist/RoadSight.app"
echo "  To run     : open dist/RoadSight.app"
echo ""
echo "User data (uploads/results) will be saved to:"
echo "  ~/Documents/RoadSight/"
