"""Step 2a: download and extract the UCI ElectricityLoadDiagrams20112014 dataset.

Downloads the official zip, extracts LD2011_2014.txt into data/raw/.
Safe to re-run: skips work already done.
"""
from pathlib import Path
import zipfile
import urllib.request

from src.config import ROOT

URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
ZIP_PATH = ROOT / "data" / "raw" / "electricity.zip"
TXT_PATH = ROOT / "data" / "raw" / "LD2011_2014.txt"


def download() -> Path:
    if TXT_PATH.exists():
        print(f"[load] already extracted: {TXT_PATH} ({TXT_PATH.stat().st_size/1e6:.0f} MB)")
        return TXT_PATH
    if not ZIP_PATH.exists():
        print(f"[load] downloading {URL} ...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print(f"[load] downloaded zip: {ZIP_PATH.stat().st_size/1e6:.0f} MB")
    print("[load] extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(ZIP_PATH.parent)
    print(f"[load] extracted: {TXT_PATH} ({TXT_PATH.stat().st_size/1e6:.0f} MB)")
    return TXT_PATH


if __name__ == "__main__":
    download()
