#!/bin/bash
# setup.sh — Urban-Eye one-command environment setup
# Usage: bash setup.sh

set -e

echo ""
echo "⬡ Urban-Eye — Environment Setup"
echo "──────────────────────────────────"

# 1. Write requirements.txt right here — no need to copy it separately
cat > requirements.txt << 'REQS'
ultralytics==8.3.0
opencv-python==4.10.0.84
flask==3.0.3
flask-cors==4.0.1
numpy==1.26.4
REQS
echo "→ requirements.txt created"

# 2. Remove old venv if it exists so we start clean
if [ -d "venv" ]; then
  echo "→ Removing old venv..."
  rm -rf venv
fi

# 3. Create fresh virtual environment
echo "→ Creating virtual environment..."
python3 -m venv venv
echo "  ✓ venv created"

# 4. Activate and upgrade pip silently
source venv/bin/activate
echo "→ Upgrading pip..."
pip install --upgrade pip --quiet
echo "  ✓ pip upgraded"

# 5. Install all dependencies
echo "→ Installing dependencies (this takes 2-3 minutes)..."
pip install -r requirements.txt
echo "  ✓ Dependencies installed"

# 6. Verify ultralytics
echo "→ Verifying YOLOv8..."
python3 -c "from ultralytics import YOLO; print('  ✓ YOLOv8 ready')"

# 7. Verify Flask
echo "→ Verifying Flask..."
python3 -c "from flask import Flask; print('  ✓ Flask ready')"

# 8. Verify OpenCV
echo "→ Verifying OpenCV..."
python3 -c "import cv2; print('  ✓ OpenCV ready')"

echo ""
echo "──────────────────────────────────"
echo "✓ Setup complete. Urban-Eye is ready."
echo ""
echo "To run — open 2 terminal tabs in this folder:"
echo ""
echo "  Tab 1:  source venv/bin/activate && python3 detect.py"
echo "  Tab 2:  source venv/bin/activate && python3 app.py"
echo "  Browser: open dashboard.html"
echo ""