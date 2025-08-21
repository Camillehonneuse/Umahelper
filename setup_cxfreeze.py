# setup_cxfreeze.py
from cx_Freeze import setup, Executable
from pathlib import Path
import sys

APP_NAME = "UmaHelper"
ENTRY_SCRIPT = "main.py"  


include_files = []
for folder in ("assets", "roi"):
    p = Path(folder)
    if p.exists():
        include_files.append((str(p), str(p)))  # (source, target)

# Modules to make sure CX_Freeze includes
includes = [
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "cv2",
    "paddleocr",
    "paddlepaddle",
    "rapidfuzz",
    "numpy",
    "mss",
    "PIL",           # Pillow
]

build_exe_options = {
    "includes": includes,
    "include_files": include_files,
    "include_msvcr": True,       # bundle VC runtimes on Windows
    "zip_include_packages": [],
    "zip_exclude_packages": ["*"],

}

base = "Win32GUI" if sys.platform == "win32" else None  # no console window

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
