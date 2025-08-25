# setup_cxfreeze.py
# Hardened cx_Freeze setup for UmaHelper
# - Uses headless OpenCV (avoid cv2 hook crashes)
# - Bundles Paddle native DLLs and PySide6 plugins
# - Includes your data folders (assets/, roi/)
# - Optionally bundles PaddleOCR models from paddle_models/

import os
import sys
from pathlib import Path
from cx_Freeze import setup, Executable

# -------- Config --------
APP_NAME = os.environ.get("APP_NAME", "UmaHelper")
ENTRY_SCRIPT = os.environ.get("ENTRY_SCRIPT", "main.py")

root = Path(__file__).resolve().parent

# -------- Data folders you want alongside the EXE --------
# Include paddle_models/ if you've pre-downloaded models there
include_files = []
for folder in ("assets", "roi", "paddle_models"):
    p = root / folder
    if p.exists():
        include_files.append((str(p), folder))  # (source, target)

# -------- PySide6 plugins (ensure 'platforms/windows' etc. are present) --------
try:
    from PySide6.QtCore import QLibraryInfo
    plugins_dir = QLibraryInfo.path(QLibraryInfo.PluginsPath)
    if plugins_dir and Path(plugins_dir).exists():
        include_files.append((plugins_dir, "PySide6/plugins"))
except Exception:
    pass

# -------- Paddle native DLLs (CRITICAL) --------
try:
    import paddle  # importable module name (pip package is 'paddlepaddle')
    libs_dir = Path(paddle.__file__).parent / "libs"
    if libs_dir.exists():
        include_files.append((str(libs_dir), "paddle/libs"))
except Exception:
    pass

# -------- Manually vendor OpenCV to avoid cx_Freeze cv2 hook --------
# We exclude 'cv2' from package scanning and copy the installed cv2 package into build/lib/cv2
cv2_vendored = False
try:
    import cv2  # should be opencv-python-headless
    cv2_dir = Path(cv2.__file__).parent
    if cv2_dir.exists():
        include_files.append((str(cv2_dir), "lib/cv2"))
        cv2_vendored = True
except Exception:
    pass
print(f"[setup] Vendored cv2: {cv2_vendored}")

# -------- Packages (NO 'includes' to avoid finder bugs) --------
packages = [
    "PySide6",
    "shiboken6",
    # "cv2",  # intentionally excluded; we vendored it via include_files
    "paddle",
    "paddleocr",
    "rapidfuzz",
    "numpy",
    "mss",
    "PIL",  # Pillow
]

# Exclude to keep cx_Freeze from walking cv2 (problematic hooks)
excludes = ["tkinter", "cv2", "cv2.*"]

build_exe_options = {
    "packages": packages,
    "include_files": include_files,
    "include_msvcr": True,
    "zip_include_packages": [],
    "zip_exclude_packages": ["*"],
    "excludes": excludes,
    # "optimize": 2,  # optional
}

# Keep console while stabilizing; switch to "Win32GUI" once you're happy
base = None  # or "Win32GUI" for no console window

setup(
    name=APP_NAME,
    version="1.0.0",
    description="UmaHelper",
    options={"build_exe": build_exe_options},
    executables=[Executable(ENTRY_SCRIPT, base=base, target_name=f"{APP_NAME}.exe")],
)
