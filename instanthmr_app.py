#!/usr/bin/env python3
"""InstantHMR v2 - Unified Studio & CLI Runner.

Usage:
    ./instanthmr_app.py -i input_video.mp4
    ./instanthmr_app.py -i input_video.mp4 --mode compare --detector yolo26s --compare-with rfdetr_nano
    ./instanthmr_app.py -i input_video.mp4 --no-smooth
    ./instanthmr_app.py -i 0  # Webcam stream
"""
import sys
from pathlib import Path

# Ensure instanthmr package is discoverable
REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from instanthmr.cli import main

if __name__ == "__main__":
    main()
