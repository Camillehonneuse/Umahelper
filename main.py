# pip install mss pillow pytesseract rapidfuzz PySide6 numpy opencv-python pywin32

import sys, os, shutil, ctypes, json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import win32gui, win32con
import mss
from PIL import Image
import pytesseract
from rapidfuzz import fuzz
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, QTimer, QRect
from PySide6.QtGui import QPainter, QPen, QColor

import cv2
import numpy as np

# =========================
# Settings (tweak here)
# =========================
GAME_TITLE_HINT   = "umamusume"
EXPECTED_ASPECT   = 16/9
TICK_MS           = 150

# Tuning for fuzzy match — start ~78–86; we’ll default to 80 with better OCR
EVENT_FUZZY_SCORE = 80

# Show ROI rectangles in a separate overlay
DEBUG             = True

# Horizontal-only padding on text ROIs (px)
PAD_X_LABEL = 8
PAD_X_TITLE = 10

# Manual pixel offsets (your calibrated values)
MANUAL_X_OFFSET = -80
MANUAL_Y_OFFSET = -50

# Save preprocessed title image for inspection (in ./debug/)
SAVE_PREPROC_TITLE = False

ROOT = Path(__file__).resolve().parent
ROI_FILE = ROOT / "roi" / "capture_areas.json"
ASSETS_DIR = ROOT / "assets"
DEBUG_DIR = ROOT / "debug"
DEBUG_DIR.mkdir(exist_ok=True)

# =========================
# DPI awareness (fixes ROI shift)
# =========================
def enable_dpi_awareness():
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # Per-monitor v2
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor v1
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()       # System DPI
    except Exception:
        pass

# =========================
# Tesseract auto-detect (Windows)
# =========================
def configure_tesseract():
    if shutil.which("tesseract"):
        return
    for exe in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(exe):
            pytesseract.pytesseract.tesseract_cmd = exe
            return
    print("[UmaHelper] ⚠️ Tesseract not found (install or add to PATH).", file=sys.stderr)

# =========================
# Window & geometry helpers
# =========================
def find_game_hwnd() -> Optional[int]:
    want = GAME_TITLE_HINT.lower()
    best = {"hwnd": None, "area": -1}
    def enum_cb(h, _):
        try:
            if not win32gui.IsWindowVisible(h):
                return
            title = win32gui.GetWindowText(h)
            if not title:
                return
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
    return sx, sy, Rc - Lc, Bc - Tc  # x,y,w,h in screen pixels

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
    if w <= 0 or h <= 0:
        return None
    s = _mss.grab({"left": l, "top": t, "width": w, "height": h})
    return Image.frombytes("RGB", s.size, s.bgra, "raw", "BGRX")

# =========================
# OCR — tight B/W pipeline for tiny white text on solid bg
# =========================
def preprocess_bw(pil_img: Image.Image, scale: float = 4.0, invert: bool = True) -> Image.Image:
    """Upscale → gray → Otsu B/W → optional invert → PIL."""
    cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if scale != 1.0:
        cv = cv2.resize(cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    # Otsu threshold (handles both red/blue backgrounds)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        # We want black text on white background for Tesseract; title is white text, so invert.
        bw = cv2.bitwise_not(bw)
    return Image.fromarray(bw)

def ocr_with_whitelist(pil_img: Image.Image, whitelist: str, psm: int = 7) -> str:
    cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
    txt = pytesseract.image_to_string(pil_img, lang="eng", config=cfg)
    return " ".join(txt.split())

def ocr_event_label(img_pil: Image.Image) -> str:
    # Less punctuation needed here, keep it simple
    bw = preprocess_bw(img_pil, scale=3.0, invert=True)
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
    text = ocr_with_whitelist(bw, whitelist, psm=7)
    if DEBUG:
        print(f"[DBG] LABEL: {repr(text)}")
    return text

def ocr_title(img_pil: Image.Image) -> str:
    # Title needs punctuation and digits; upscale harder for tiny text
    bw = preprocess_bw(img_pil, scale=4.0, invert=True)
    if SAVE_PREPROC_TITLE:
        try:
            bw.save(DEBUG_DIR / "title_preproc.png")
        except Exception:
            pass
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 !?':-"
    text = ocr_with_whitelist(bw, whitelist, psm=7)
    if DEBUG:
        print(f"[DBG] TITLE: {repr(text)}")
    return text

def normalize_title(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    return " ".join(s.split())

# =========================
# Data loading (global index)
# =========================
def load_rois() -> Dict[str, Any]:
    with open(ROI_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("areas", data)

def build_global_event_index() -> Dict[str, List[Dict[str, Any]]]:
    """
    title/variants (normalized) -> list of entries:
    { "type": "card"/"trainee", "actor": "AgnesTachyon", "event": {...}, "events_path": str }
    """
    index: Dict[str, List[Dict[str, Any]]] = {}
    for kind in ["cards", "trainees"]:
        base = ASSETS_DIR / kind
        if not base.exists():
            continue
        for actor_dir in base.iterdir():
            if not actor_dir.is_dir():
                continue
            evfile = actor_dir / "events.json"
            if not evfile.exists():
                continue
            try:
                with open(evfile, "r", encoding="utf-8") as f:
                    events = json.load(f).get("events", [])
            except Exception:
                continue
            for ev in events:
                titles = [ev.get("title", "")]
                titles += ev.get("title_variants", [])
                for t in titles:
                    key = normalize_title(t)
                    if not key:
                        continue
                    index.setdefault(key, []).append({
                        "type": "card" if kind == "cards" else "trainee",
                        "actor": actor_dir.name,
                        "event": ev,
                        "events_path": str(evfile)
                    })
    print(f"[UmaHelper] Indexed titles: {sum(len(v) for v in index.values())}")
    return index

def find_global_event_by_title(global_idx: Dict[str, List[Dict[str, Any]]],
                               ocr_title_str: str,
                               min_score: int = EVENT_FUZZY_SCORE) -> Optional[Dict[str, Any]]:
    """
    Fuzzy-match OCR title across all known titles.
    Returns {"type","actor","event","score","best_key"} or None.
    Also prints Top-5 candidates for tuning.
    """
    q = normalize_title(ocr_title_str)
    if not q:
        if DEBUG: print("[DBG] OCR title empty after normalization.")
        return None

    top5: List[Tuple[int, str, Dict[str, Any]]] = []
    best = None
    best_score = -1
    best_key = ""

    for key, entries in global_idx.items():
        score = fuzz.token_sort_ratio(q, key)
        top5.append((int(score), key, entries[0]))
        if score > best_score:
            best = entries[0]
            best_score = score
            best_key = key

    if DEBUG:
        top5.sort(key=lambda x: x[0], reverse=True)
        print(f"[DBG] OCR Title='{ocr_title_str}' | norm='{q}'")
        print("[DBG] Top matches:")
        for s, k, e in top5[:5]:
            print(f"      {s:>3}  {k}   ->   {e['type']}::{e['actor']}::{e['event'].get('title','')}")

    if best and best_score >= min_score:
        return {"type": best["type"], "actor": best["actor"], "event": best["event"], "score": int(best_score), "best_key": best_key}
    return None

# =========================
# Overlays (info + debug ROIs)
# =========================
class InfoOverlay(QWidget):
    """Answer overlay with black semi-transparent background."""
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # Click-through
        hwnd = int(self.winId())
        ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(14, 12, 14, 12)
        self.labels: List[QLabel] = []

        self.setStyleSheet("""
            QWidget { background-color: rgba(0,0,0,190); border-radius: 8px; }
            QLabel  { color: white; font: 18px 'Segoe UI'; }
        """)
        self.resize(560, 180)

    def set_lines(self, lines: List[str]):
        for w in self.findChildren(QLabel):
            self.layout.removeWidget(w)
            w.deleteLater()
        for ln in lines:
            lab = QLabel(ln)
            self.layout.addWidget(lab)
            self.labels.append(lab)
        self.adjustSize()

class DebugOverlay(QWidget):
    """Transparent overlay to draw ROI rectangles (never passed to OCR)."""
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # Click-through
        hwnd = int(self.winId())
        ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_LAYERED)
        self.client_origin = (0, 0)
        self.rects: List[Tuple[str, QRect, QColor]] = []

    def set_client_geometry(self, x: int, y: int, w: int, h: int):
        self.setGeometry(x, y, w, h)
        self.client_origin = (x, y)

    def set_rois(self, roi_map: Dict[str, Tuple[int,int,int,int]]):
        colors = {
            "event_type_label": QColor(0, 255, 0, 200),
            "title_text": QColor(255, 215, 0, 200),
        }
        self.rects.clear()
        cx, cy = self.client_origin
        for name, (l, t, r, b) in roi_map.items():
            qrect = QRect(l - cx, t - cy, r - l, b - t)
            self.rects.append((name, qrect, colors.get(name, QColor(255, 0, 0, 200))))
        self.update()

    def paintEvent(self, _ev):
        if not self.rects:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        for name, rect, color in self.rects:
            pen = QPen(color); pen.setWidth(3); p.setPen(pen)
            p.drawRect(rect)
            p.setPen(QPen(QColor(255,255,255,220)))
            label_y = rect.y()-6 if rect.y() > 12 else rect.y()+14
            p.drawText(rect.x()+4, label_y, name)
        p.end()

# =========================
# Main app
# =========================
class UmaHelperApp:
    def __init__(self):
        self.hwnd = find_game_hwnd()
        if not self.hwnd:
            raise RuntimeError("Umamusume window not found.")
        self.rois = load_rois()
        self.global_index = build_global_event_index()

        self.info = InfoOverlay(); self.info.show()
        self.debug = DebugOverlay() if DEBUG else None
        if self.debug: self.debug.show()

    def tick(self):
        hwnd = find_game_hwnd()
        if hwnd:
            self.hwnd = hwnd  # refresh if user refocused window

        cx, cy, cw, ch = get_client_rect_screen(self.hwnd)
        if self.debug:
            self.debug.set_client_geometry(cx, cy, cw, ch)

        # Resolve pixel ROIs + apply horizontal-only padding
        r_label = pad_rect_xy(roi_to_pixels(self.rois["event_type_label"], cx, cy, cw, ch), PAD_X_LABEL, 0)
        r_title = pad_rect_xy(roi_to_pixels(self.rois["title_text"],       cx, cy, cw, ch), PAD_X_TITLE, 0)

        if self.debug:
            # draw the original (non-padded) boxes for visual reference
            self.debug.set_rois({
                "event_type_label": roi_to_pixels(self.rois["event_type_label"], cx, cy, cw, ch),
                "title_text":       roi_to_pixels(self.rois["title_text"],       cx, cy, cw, ch),
            })

        # Grab ROIs (raw)
        img_label = grab_region(r_label)
        img_title = grab_region(r_title)
        if not img_label or not img_title:
            self.info.set_lines(["Waiting for screen…"])
            return

        # OCR
        label_txt = ocr_event_label(img_label)   # helpful for display
        title_txt = ocr_title(img_title)

        # Lookup by title (fuzzy)
        hit = find_global_event_by_title(self.global_index, title_txt, min_score=EVENT_FUZZY_SCORE)

        if not hit:
            lines = [
                f"Type: {label_txt or '(?)'}",
                f"Title: {title_txt or '(…)' }",
                f"No match ≥ {EVENT_FUZZY_SCORE}. Check console Top-5.",
            ]
            self.info.set_lines(lines)
        else:
            ev = hit["event"]
            header = f"{hit['actor']} ({hit['type']}) · {ev.get('title','')}  [score {hit['score']}]"
            lines = [header, "— Choices —"]
            # compact effects line
            short_map = {"speed":"SPD","stamina":"STA","power":"POW","guts":"GUTS","wisdom":"WIS","skill_points":"SP","bond":"BOND","energy":"ENG"}
            for ch in ev.get("choices", []):
                lab = ch.get("label", "Choice")
                eff = ch.get("effects", {})
                parts = []
                for k, v in eff.items():
                    code = short_map.get(k, k.upper())
                    sign = "+" if isinstance(v, (int, float)) and v >= 0 else ""
                    parts.append(f"{code}{sign}{v}")
                lines.append(f"{lab}  |  {', '.join(parts)}")
            self.info.set_lines(lines)

        # Position info overlay at ~1/3 from left, middle of viewport
        vx, vy, vw, vh = compute_viewport(cw, ch)
        anchor_x = cx + vx + int(vw * 0.3333)
        anchor_y = cy + vy + int(vh * 0.5)
        ox = anchor_x - self.info.width() // 2
        oy = anchor_y - self.info.height() // 2
        ox = max(cx, min(ox, cx + cw - self.info.width()))
        oy = max(cy, min(oy, cy + ch - self.info.height()))
        self.info.move(ox, oy)

# =========================
# Entry point
# =========================
def main():
    enable_dpi_awareness()
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    configure_tesseract()

    app = QApplication(sys.argv)
    helper = UmaHelperApp()

    timer = QTimer()
    timer.timeout.connect(helper.tick)
    timer.start(TICK_MS)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
