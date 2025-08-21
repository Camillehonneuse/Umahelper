import os
import sys
from pathlib import Path
from cx_Freeze import setup, Executable

APP_NAME = "UmaHelper"
ENTRY_SCRIPT = os.environ.get("ENTRY_SCRIPT", "main.py")  # set in CI if needed

# Bundle your data folders
include_files = []
for folder in ("assets", "roi"):
    p = Path(folder)
    if p.exists():
        include_files.append((str(p), str(p)))

# IMPORTANT:
# Use "packages" (whole packages) instead of "includes" to avoid the finder bug.
packages = [
    "PySide6",
    "shiboken6",
    "cv2",
    "paddleocr",
    "paddlepaddle",
    "rapidfuzz",
    "numpy",
    "mss",
    "PIL",          
    "json",        
]

build_exe_options = {
    "packages": packages,
    "include_files": include_files,
    "include_msvcr": True,
    # Keep libraries unzipped; Qt/Paddle/OpenCV prefer loose files
    "zip_include_packages": [],
    "zip_exclude_packages": ["*"],
    # If you want a smaller build, you can exclude tkinter, tests, etc.
    # "excludes": ["tkinter"],
}

# GUI base on Windows (no console). Change to None to keep a console for logs.
base = "Win32GUI" if sys.platform == "win32" else None

executables = [
    Executable(
        script=ENTRY_SCRIPT,
        base=base,
        target_name=f"{APP_NAME}.exe",
    )
]

setup(
    name=APP_NAME,
    version="1.0.0",
    description="UmaHelper",
    options={"build_exe": build_exe_options},
    executables=executables,
)
