# UmaHelper - OCR (label+title) → global lookup (cards + common + legacy) → overlay
# Deps: pip install mss pillow pytesseract rapidfuzz PySide6 numpy opencv-python pywin32

import sys, os, shutil, ctypes, json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import win32gui, win32con
import mss
from PIL import Image
import pytesseract
from rapidfuzz import fuzz
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, QPoint
import cv2
import numpy as np

# =========================
# Settings
# =========================
GAME_TITLE_HINT   = "umamusume"
EXPECTED_ASPECT   = 16/9

EVENT_FUZZY_SCORE = 80
PAD_X_LABEL = 8
PAD_X_TITLE = 10
MANUAL_X_OFFSET = -80
MANUAL_Y_OFFSET = -50

DEBUG_LOG = False

ROOT        = Path(__file__).resolve().parent
ROI_FILE    = ROOT / "roi" / "capture_areas.json"
ASSETS_DIR  = ROOT / "assets"
CARDS_DIR   = ASSETS_DIR / "cards"
TRAINEES_DIR= ASSETS_DIR / "trainees"
COMMON_FILE = ASSETS_DIR / "common" / "events.json"

# =========================
# DPI awareness
# =========================
def enable_dpi_awareness():
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# =========================
# Tesseract setup
# =========================
def configure_tesseract():
    if shutil.which("tesseract"): return
    for exe in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(exe):
            pytesseract.pytesseract.tesseract_cmd = exe
            return
    print("[UmaHelper] ⚠️ Tesseract not found (install or add to PATH).", file=sys.stderr)

# =========================
# Window & geometry
# =========================
def find_game_hwnd() -> Optional[int]:
    want = GAME_TITLE_HINT.lower()
    best = {"hwnd": None, "area": -1}
    def enum_cb(h, _):
        try:
            if not win32gui.IsWindowVisible(h): return
            title = win32gui.GetWindowText(h) or ""
            if want in title.lower():
                L, T, R, B = win32gui.GetClientRect(h)
                area = max(0, (R - L) * (B - T))
                if area > best["area"]:
                    best["hwnd"] = h
                    best["area"] = area
        except Exception:
            pass
    win32gui.EnumWindows(enum_cb, None)
    return best["hwnd"]

def get_client_rect_screen(hwnd: int) -> Tuple[int, int, int, int]:
    Lc, Tc, Rc, Bc = win32gui.GetClientRect(hwnd)
    sx, sy = win32gui.ClientToScreen(hwnd, (0, 0))
    return sx, sy, Rc - Lc, Bc - Tc

def compute_viewport(client_w: int, client_h: int, expect_aspect: float = EXPECTED_ASPECT):
    vw = client_w
    vh = int(vw / expect_aspect)
    if vh > client_h:
        vh = client_h
        vw = int(vh * expect_aspect)
    vx = (client_w - vw) // 2
    vy = (client_h - vh) // 2
    return vx, vy, vw, vh

def roi_to_pixels(norm_roi: List[float], cx: int, cy: int, cw: int, ch: int):
    vx, vy, vw, vh = compute_viewport(cw, ch)
    l = cx + vx + int(norm_roi[0] * vw)
    t = cy + vy + int(norm_roi[1] * vh)
    r = cx + vx + int(norm_roi[2] * vw)
    b = cy + vy + int(norm_roi[3] * vh)
    return (l + MANUAL_X_OFFSET, t + MANUAL_Y_OFFSET,
            r + MANUAL_X_OFFSET, b + MANUAL_Y_OFFSET)

def pad_rect_xy(rect: Tuple[int,int,int,int], pad_x: int = 0, pad_y: int = 0):
    l, t, r, b = rect
    return (l - pad_x, t - pad_y, r + pad_x, b + pad_y)

# =========================
# Capture
# =========================
_mss = mss.mss()
def grab_region(region: Tuple[int, int, int, int]) -> Optional[Image.Image]:
    l, t, r, b = region
    w, h = r - l, b - t
    if w <= 0 or h <= 0: return None
    s = _mss.grab({"left": l, "top": t, "width": w, "height": h})
    return Image.frombytes("RGB", s.size, s.bgra, "raw", "BGRX")

# =========================
# OCR helpers
# =========================
def preprocess_bw(pil_img: Image.Image, scale: float = 4.0, invert: bool = True) -> Image.Image:
    cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if scale != 1.0:
        cv = cv2.resize(cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        bw = cv2.bitwise_not(bw)
    return Image.fromarray(bw)

def ocr_with_whitelist(pil_img: Image.Image, whitelist: str, psm: int = 7) -> str:
    cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
    txt = pytesseract.image_to_string(pil_img, lang="eng", config=cfg)
    return " ".join(txt.split())

def ocr_title(img_pil: Image.Image) -> str:
    bw = preprocess_bw(img_pil, scale=4.0, invert=True)
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 !?':-"
    return ocr_with_whitelist(bw, whitelist, psm=7)

# =========================
# Matching helpers
# =========================
def normalize_title(s: str) -> str:
    s = s.strip()
    trans = str.maketrans({"—":"-","–":"-","“":"\"","”":"\"","‘":"'","’":"'"})
    s = s.translate(trans)
    s = "".join(ch for ch in s if (ch.isalnum() or ch.isspace() or ch in "-!'?"))
    return " ".join(s.split()).lower()

def compact(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum()).lower()

def find_event_by_title(global_idx: Dict[str, Dict[str, Any]],
                        ocr_title_str: str,
                        min_score: int = EVENT_FUZZY_SCORE):
    q = normalize_title(ocr_title_str)
    if not q:
        return None, None
    q_c = compact(q)

    best_key, best_score = None, -1
    for key in global_idx.keys():
        s1 = fuzz.token_sort_ratio(q, key)
        s2 = fuzz.partial_ratio(q, key)
        s3 = fuzz.QRatio(q, key)
        s4 = fuzz.ratio(q_c, compact(key))
        score = max(s1, s2, s3, s4)
        if score > best_score:
            best_key, best_score = key, score

    if best_key:
        compact_ok = fuzz.ratio(q_c, compact(best_key)) >= 90
        if best_score >= min_score or compact_ok:
            return global_idx[best_key], int(best_score)

    return None, int(best_score)

# =========================
# Data loading (unchanged from your version)
# =========================
def load_rois():
    with open(ROI_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("areas", data)

def build_global_event_index():
    index = {}
    if COMMON_FILE.exists():
        try:
            common = json.load(open(COMMON_FILE, "r", encoding="utf-8"))
            for ev in common.get("common_events", []):
                title = ev.get("event_name", "").strip()
                key = normalize_title(title)
                if key:
                    index[key] = {"title": title, "origin": "common", "choices": []}
        except: pass
    if CARDS_DIR.exists():
        for jf in CARDS_DIR.glob("*.json"):
            if jf.name.lower() in ("events.json","common.json"): 
                continue
            try:
                data = json.load(open(jf, "r", encoding="utf-8"))
                for ev in data.get("events", []):
                    title = ev.get("name", "").strip()
                    key = normalize_title(title)
                    if key:
                        index[key] = {"title": title, "origin": jf.stem, "choices": []}
            except: pass
    return index

# =========================
# Overlay (now draggable + scan button)
# =========================
class InfoOverlay(QWidget):
    def __init__(self, scan_callback):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        hwnd = int(self.winId())
        ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                               ex | win32con.WS_EX_LAYERED)

        self.drag_pos = None
        self.scan_callback = scan_callback

        # Layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Header with scan button
        header_layout = QHBoxLayout()
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self.scan_callback)
        header_layout.addWidget(self.scan_btn)
        header_layout.addStretch()
        outer.addLayout(header_layout)

        # Info lines
        self.info_layout = QVBoxLayout()
        outer.addLayout(self.info_layout)

        self.setStyleSheet("""
            QWidget { background-color: rgba(0,0,0,200); border-radius: 8px; }
            QLabel  { color: white; font: 18px 'Segoe UI'; }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)

    def set_lines(self, lines: List[str]):
        for w in self.findChildren(QLabel):
            self.info_layout.removeWidget(w)
            w.deleteLater()
        for ln in lines:
            self.info_layout.addWidget(QLabel(ln))
        self.adjustSize()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_pos is not None:
            self.move(event.globalPos() - self.drag_pos)
            event.accept()

# =========================
# Main app
# =========================
class UmaHelperApp:
    def __init__(self):
        self.hwnd = find_game_hwnd()
        if not self.hwnd:
            raise RuntimeError("Umamusume window not found.")
        self.rois = load_rois()
        self.index = build_global_event_index()
        self.overlay = InfoOverlay(self.scan)
        self.overlay.set_lines(["Searching..."])
        self.overlay.show()

    def scan(self):
        cx, cy, cw, ch = get_client_rect_screen(self.hwnd)
        r_title = pad_rect_xy(roi_to_pixels(self.rois["title_text"], cx, cy, cw, ch), PAD_X_TITLE, 0)
        img_title = grab_region(r_title)
        if not img_title:
            self.overlay.set_lines(["Searching..."])
            return

        title_txt = ocr_title(img_title)
        ev, score = find_event_by_title(self.index, title_txt, min_score=EVENT_FUZZY_SCORE)

        if not ev:
            self.overlay.set_lines([f"No match ≥ {EVENT_FUZZY_SCORE}", f"Title: {title_txt or '(...)'}"])
        else:
            header = f"{ev['title']}  [{ev['origin']}, score {score}]"
            lines = [header]
            self.overlay.set_lines(lines)

# =========================
# Entry
# =========================
def main():
    enable_dpi_awareness()
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    configure_tesseract()

    app = QApplication(sys.argv)
    helper = UmaHelperApp()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
