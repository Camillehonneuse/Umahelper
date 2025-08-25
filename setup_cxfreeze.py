# setup_cxfreeze.py
import os, sys
from pathlib import Path
from cx_Freeze import setup, Executable

APP_NAME = "UmaHelper"
ENTRY_SCRIPT = os.environ.get("ENTRY_SCRIPT", main.py")  # change or set in CI

root = Path(__file__).resolve().parent

include_files = []
for folder in ("assets", "roi", "paddle_models"):  # paddle_models only if you pre-downloaded models
    p = root / folder
    if p.exists():
        include_files.append((str(p), folder))  # (source, target)

# PySide6 plugins
try:
    from PySide6.QtCore import QLibraryInfo
    plugins_dir = QLibraryInfo.path(QLibraryInfo.PluginsPath)
    if plugins_dir and Path(plugins_dir).exists():
        include_files.append((plugins_dir, "PySide6/plugins"))
except Exception:
    pass

# Paddle native DLLs
try:
    import paddle
    libs_dir = Path(paddle.__file__).parent / "libs"
    if libs_dir.exists():
        include_files.append((str(libs_dir), "paddle/libs"))
except Exception:
    pass

# Manually vendor OpenCV (we recommend using opencv-python-headless)
cv2_added = False
try:
    import cv2
    cv2_dir = Path(cv2.__file__).parent
    if cv2_dir.exists():
        include_files.append((str(cv2_dir), "lib/cv2"))
        cv2_added = True
except Exception:
    pass
print(f"[setup] Vendored cv2: {cv2_added}")

# ONLY packages (no 'includes'!)
packages = [
    "PySide6",
    "shiboken6",
    # "cv2",  # excluded from packages; we vendor it via include_files above
    "paddle",       # <-- CORRECT import name
    "paddleocr",
    "rapidfuzz",
    "numpy",
    "mss",
    "PIL",
]

excludes = ["tkinter", "cv2", "cv2.*"]  # prevent cv2 hook from running

build_exe_options = {
    "packages": packages,
    "include_files": include_files,
    "include_msvcr": True,
    "zip_include_packages": [],
    "zip_exclude_packages": ["*"],
    "excludes": excludes,
}

# Keep console for clear error messages while stabilizing
base = None  # change to "Win32GUI" when you're happy

setup(
    name=APP_NAME,
    version="1.0.0",
    description="UmaHelper",
    options={"build_exe": build_exe_options},
    executables=[Executable(ENTRY_SCRIPT, base=base, target_name=f"{APP_NAME}.exe")],
)
