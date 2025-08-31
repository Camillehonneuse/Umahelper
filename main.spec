# -*- mode: python ; coding: utf-8 -*-
import os
import site
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files
from PyInstaller.building.build_main import Tree

block_cipher = None

# ----- Paths -----
site_pkgs = Path(site.getsitepackages()[0])
cython_utility_dir = site_pkgs / "Cython" / "Utility"

# ----- App data you want to ship -----
datas = [
    ('assets', 'assets'),
    ('roi', 'roi'),
]

# Ship the entire Cython/Utility directory (fixes CppSupport.cpp, etc.)
if cython_utility_dir.exists():
    datas.append(Tree(str(cython_utility_dir), prefix='Cython/Utility'))
else:
    # Fallback: still try to collect data files if Tree path somehow missing
    datas += collect_data_files('Cython', includes=['Utility/*'])

# ----- Collect EVERYTHING for the dynamic/import-heavy libs -----
# PaddleOCR & PaddlePaddle
po_datas, po_bins, po_hidden = collect_all('paddleocr')
pd_datas, pd_bins, pd_hidden = collect_all('paddle')

# Native extension used by PaddleOCR postprocess
pc_datas, pc_bins, pc_hidden = collect_all('pyclipper')

# scikit-image stack + deps
sk_datas, sk_bins, sk_hidden = collect_all('skimage')
io_datas, io_bins, io_hidden = collect_all('imageio')
nw_datas, nw_bins, nw_hidden = collect_all('networkx')
tf_datas, tf_bins, tf_hidden = collect_all('tifffile')
pw_datas, pw_bins, pw_hidden = collect_all('PyWavelets')
sp_datas, sp_bins, sp_hidden = collect_all('scipy')

# imgaug (augmentation)
ia_datas, ia_bins, ia_hidden = collect_all('imgaug')

# yaml + shapely (configs + geometry ops)
ya_datas, ya_bins, ya_hidden = collect_all('yaml')
sh_datas, sh_bins, sh_hidden = collect_all('shapely')

# lmdb (dataset backend)
lm_datas, lm_bins, lm_hidden = collect_all('lmdb')

# Merge
datas += (po_datas + pd_datas + pc_datas +
          sk_datas + io_datas + nw_datas + tf_datas + pw_datas + sp_datas +
          ia_datas + ya_datas + sh_datas + lm_datas)

binaries = (po_bins + pd_bins + pc_bins +
            sk_bins + io_bins + nw_bins + tf_bins + pw_bins + sp_bins +
            ia_bins + ya_bins + sh_bins + lm_bins)

hiddenimports = list(set(po_hidden + pd_hidden + pc_hidden +
                         sk_hidden + io_hidden + nw_hidden + tf_hidden + pw_hidden + sp_hidden +
                         ia_hidden + ya_hidden + sh_hidden + lm_hidden))

# Force dynamic/stdlib modules PaddleOCR reaches for at runtime
forced_hidden = [
    'imghdr',
    'skimage.morphology._skeletonize',
    'skimage.filters._gaussian',
    'yaml',
]
hiddenimports += [m for m in forced_hidden if m not in hiddenimports]

a = Analysis(
    ['main.py'],              # your entry point
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='UmaHelper',         # exe name
    debug=False,               # keep for now
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False           # visible console for logs
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='UmaHelper'          # onedir folder name
)
