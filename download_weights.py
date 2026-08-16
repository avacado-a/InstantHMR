#!/usr/bin/env python3
"""
Download MHR Model Weights
==========================
Downloads Meta's official MHR model weights (mhr_model.pt) and places it into assets/.
"""

import os
import sys
import zipfile
import urllib.request

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
TARGET_FILE = os.path.join(ASSETS_DIR, "mhr_model.pt")
MHR_RELEASE_URL = "https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"

def download_mhr_weights():
    if os.path.exists(TARGET_FILE):
        print(f"✓ MHR model weights already exist at: {TARGET_FILE}")
        return

    os.makedirs(ASSETS_DIR, exist_ok=True)
    temp_zip = os.path.join(REPO_ROOT, "mhr_temp.zip")

    print(f"Downloading official MHR weights from: {MHR_RELEASE_URL}")
    print("Please wait...")

    try:
        urllib.request.urlretrieve(MHR_RELEASE_URL, temp_zip)
        print("Extracting mhr_model.pt to assets/...")

        with zipfile.ZipFile(temp_zip, "r") as zf:
            if "mhr_model.pt" in zf.namelist():
                zf.extract("mhr_model.pt", ASSETS_DIR)
            else:
                for member in zf.namelist():
                    if member.endswith("mhr_model.pt"):
                        with zf.open(member) as source, open(TARGET_FILE, "wb") as target:
                            target.write(source.read())

        print(f"✅ Successfully downloaded and installed: {TARGET_FILE}")
    finally:
        if os.path.exists(temp_zip):
            os.remove(temp_zip)

if __name__ == "__main__":
    download_mhr_weights()
