# setup_cxfreeze.py
# cx_Freeze build for UmaHelper (no 'packages' forcing to avoid include_package crashes)

import os
import sys
from pathlib import Path
from cx_Freeze import setup, Executable

APP_NAME = os.environ.get("APP_NAME", "UmaHelper")
ENTRY_SCRIPT = os.environ.get("ENTRY_SCRIPT", "main.py")

root = Path(__file__).resolve().parent

# ---- Copy runtime data next to the EXE ----
include_files = []
for folder in ("assets", "roi", "paddle_models"):  # paddle_models only if you pre-downloaded
    p = root / folder
    if p.exists():
        include_files.append((str(p), folder))  # (source, target)

# ---- PySide6 plugins (Qt platforms etc.) ----
try:
    from PySide6.QtCore import QLibraryInfo
    plugins_dir = QLibraryInfo.path(QLibraryInfo.PluginsPath)
    if plugins_dir and Path(plugins_dir).exists():
        include_files.append((plugins_dir, "PySide6/plugins"))
except Exception:
    pass

# ---- Paddle native DLLs ----
try:
    import paddle  # importable module is 'paddle' (pip name is 'paddlepaddle')
    libs_dir = Path(paddle.__file__).parent / "libs"
    if libs_dir.exists():
        include_files.append((str(libs_dir), "paddle/libs"))
except Exception:
    pass

# ---- Vendor OpenCV to avoid cv2 hook ----
cv2_vendored = False
try:
    import cv2  # recommend opencv-python-headless in CI
    cv2_dir = Path(cv2.__file__).parent
    if cv2_dir.exists():
        include_files.append((str(cv2_dir), "lib/cv2"))
        cv2_vendored = True
except Exception:
    pass
print(f"[setup] Vendored cv2: {cv2_vendored}")

# ---- DO NOT set 'packages' or 'includes' ----
# Let cx_Freeze discover imports from your code to avoid include_package bugs.

excludes = [
    "tkinter",
    "cv2", "cv2.*",   # we vendor cv2 manually
]

build_exe_options = {
    "include_files": include_files,
    "include_msvcr": True,
    # keep everything unzipped for simpler loading of large libs
    "zip_include_packages": [],
    "zip_exclude_packages": ["*"],
    "excludes": excludes,
    # "optimize": 2,  # optional
}

# Keep console during stabilization so you can see runtime errors
base = None  # switch to "Win32GUI" later if you want no console window

setup(
    name=APP_NAME,
    version="1.0.0",
    description="UmaHelper",
    options={"build_exe": build_exe_options},
    executables=[Executable(ENTRY_SCRIPT, base=base, target_name=f"{APP_NAME}.exe")],
)
