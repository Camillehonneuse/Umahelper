# UmaHelper (Multi-Monitor + Split Windows, ROI overlay fix, threaded Scan)
# Deps (CPU only):
#   pip install mss pillow paddleocr rapidfuzz PySide6 numpy opencv-python

import sys, ctypes, json
from pathlib import Path
from typing import Optional, Dict, Any, List

import mss
from PIL import Image
from rapidfuzz import fuzz

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QFrame, QComboBox, QGraphicsDropShadowEffect
)
from PySide6.QtCore import (
    Qt, QRect, QPoint, Signal, QEvent, QObject, QThread, Slot
)
from PySide6.QtGui import QPainter, QColor, QPen, QMouseEvent, QGuiApplication, QFont

import cv2
import numpy as np

# =========================
# Settings / Paths
# =========================
EVENT_FUZZY_SCORE = 80
DEBUG_LOG = True  # set True for console logs

def _log(*msg):
    if DEBUG_LOG:
        print("[UmaHelper]", *msg, flush=True)

ROOT        = Path(__file__).resolve().parent
ROI_FILE    = ROOT / "roi" / "capture_areas.json"

ASSETS_DIR  = ROOT / "assets"
CARDS_DIR   = ASSETS_DIR / "cards"
TRAINEES_DIR= ASSETS_DIR / "trainees"
COMMON_FILE = ASSETS_DIR / "common" / "events.json"

DEFAULT_ROI_LOCAL = {
    "title_text":       [200, 200, 700, 250],   # l,t,r,b
    "event_type_label": [200, 160, 420, 195],
}

# =========================
# DPI awareness (Windows)
# =========================
def enable_dpi_awareness():
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)); return
    except Exception: pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2); return
    except Exception: pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception: pass

# =========================
# PaddleOCR setup (CPU, version-tolerant)
# =========================
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None  # will raise on init

_OCR = None

def _init_paddleocr() -> "PaddleOCR":
    if PaddleOCR is None:
        raise RuntimeError("paddleocr is not installed. Run: pip install paddleocr")
    import inspect
    sig = inspect.signature(PaddleOCR.__init__)
    accepts = set(sig.parameters.keys())
    kwargs = {}
    if 'lang' in accepts: kwargs['lang'] = 'en'
    if 'use_gpu' in accepts: kwargs['use_gpu'] = False
    if 'use_textline_orientation' in accepts: kwargs['use_textline_orientation'] = True
    elif 'use_angle_cls' in accepts: kwargs['use_angle_cls'] = True
    try:
        ocr = PaddleOCR(**kwargs)
    except Exception:
        try:
            ocr = PaddleOCR(lang='en')
        except Exception:
            ocr = PaddleOCR()
    return ocr

def _get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = _init_paddleocr()
        _log("PaddleOCR initialized")
    return _OCR

# =========================
# Multi-screen helpers
# =========================
def list_screens():
    out = []
    for i, s in enumerate(QGuiApplication.screens()):
        geo = s.geometry()
        label = f"Screen {i}  ({geo.width()}x{geo.height()} @{geo.x()},{geo.y()})"
        out.append((str(i), s, geo, label))
    return out

# =========================
# ROI persistence
# =========================
def _ensure_rect(rect):
    l,t,r,b = [int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])]
    if r < l: l, r = r, l
    if b < t: t, b = b, t
    return [l,t,r,b]

def load_all_rois() -> Dict[str, Dict[str, List[int]]]:
    try:
        data = json.load(open(ROI_FILE, "r", encoding="utf-8"))
    except Exception:
        data = {}
    if "screens" in data and isinstance(data["screens"], dict):
        screens = {}
        for sid, areas in data["screens"].items():
            if not isinstance(areas, dict): continue
            fixed = {}
            for k,v in areas.items():
                if isinstance(v, list) and len(v)==4:
                    fixed[k] = _ensure_rect(v)
            if fixed: screens[str(sid)] = fixed
        if screens: return screens

    areas = data.get("areas", data) if isinstance(data, dict) else {}
    fixed = {}
    for k,v in areas.items():
        if isinstance(v, list) and len(v)==4:
            fixed[k] = _ensure_rect(v)
    if fixed: return {"0": fixed}
    return {"0": {}}

def save_all_rois(screens_map: Dict[str, Dict[str, List[int]]]):
    try:
        ROI_FILE.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"screens": screens_map}, open(ROI_FILE, "w", encoding="utf-8"), indent=2)
    except Exception as e:
        print("[ROI] save error:", e, file=sys.stderr)

def default_rois_for_screen(geo: QRect) -> Dict[str, List[int]]:
    base = {}
    for k, v in DEFAULT_ROI_LOCAL.items():
        l, t, r, b = v
        base[k] = [geo.x()+l, geo.y()+t, geo.x()+r, geo.y()+b]
        bl = base[k]
        bl[0] = max(geo.x(), bl[0]); bl[1] = max(geo.y(), bl[1])
        bl[2] = min(geo.right(), bl[2]); bl[3] = min(geo.bottom(), bl[3])
    return base

# =========================
# Screen capture (absolute)
# =========================
_mss_inst = None
def grab_abs_rect(rect: List[int]) -> Optional[Image.Image]:
    global _mss_inst
    if _mss_inst is None:
        _mss_inst = mss.mss()  # lazy init
    l, t, r, b = rect
    w, h = r - l, b - t
    if w <= 0 or h <= 0: return None
    s = _mss_inst.grab({"left": l, "top": t, "width": w, "height": h})
    return Image.frombytes("RGB", s.size, s.bgra, "raw", "BGRX")

# =========================
# OCR + matching
# =========================
def _paddle_lines(pil_img: Image.Image) -> List[str]:
    """
    Run PaddleOCR and return text lines (sorted top->bottom, left->right).
    Accepts RGB or grayscale PIL images.
    """
    ocr = _get_ocr()
    img = np.array(pil_img)

    # Ensure 3-channel BGR for Paddle
    if img.ndim == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # RGBA -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:  # RGB -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    try:
        res = ocr.ocr(img)  # some versions suggest .predict(); .ocr is widely available
    except TypeError:
        res = ocr.ocr(img, det=True, rec=True)

    items = res
    if isinstance(res, list) and len(res) == 1 and isinstance(res[0], list):
        items = res[0]

    out: List[str] = []
    parsed = []
    for it in (items or []):
        try:
            box = it[0]; info = it[1]
            if isinstance(info, (list, tuple)):
                text = info[0]
                conf = float(info[1]) if len(info) > 1 and info[1] is not None else 0.0
            elif isinstance(info, str):
                text, conf = info, 1.0
            else:
                continue
            ys = [p[1] for p in box]; xs = [p[0] for p in box]
            parsed.append(((sum(ys)/4.0, sum(xs)/4.0), text, conf))
        except Exception:
            continue

    parsed.sort(key=lambda x: (x[0][0], x[0][1]))
    for _c, text, conf in parsed:
        if conf >= 0.35 and text:
            out.append(text)
    return out

def _filter_whitelist(s: str, whitelist: str) -> str:
    if not whitelist: return s
    allowed = set(whitelist)
    return "".join(ch for ch in s if ch in allowed)

def ocr_with_whitelist(pil_img: Image.Image, whitelist: str) -> str:
    lines = _paddle_lines(pil_img)
    txt = " ".join(" ".join(lines).split())
    return _filter_whitelist(txt, whitelist)

def preprocess_variant_a(pil_img: Image.Image) -> Image.Image:
    cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv = cv2.resize(cv, None, fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.bitwise_not(bw)
    return Image.fromarray(bw)

def preprocess_variant_b(pil_img: Image.Image) -> Image.Image:
    cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv = cv2.resize(cv, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 8)
    bw = cv2.bitwise_not(bw)
    kernel = np.ones((2,2), np.uint8)
    bw = cv2.dilate(bw, kernel, iterations=1)
    return Image.fromarray(bw)

def best_ocr_title(img_pil: Image.Image) -> str:
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 !?'-"
    candidates: List[str] = []
    for prep in (preprocess_variant_a, preprocess_variant_b):
        try:
            cand = ocr_with_whitelist(prep(img_pil), whitelist)
            if cand: candidates.append(cand)
        except Exception as e:
            _log("OCR preproc failed:", e)
    if not candidates:
        raw = ocr_with_whitelist(img_pil, whitelist)
        if raw: candidates.append(raw)
    if not candidates: return ""

    def score_local(t: str) -> int:
        n = len(t); bad = sum(ch in "_/\\[]" for ch in t)
        return n*2 - bad*3
    candidates.sort(key=score_local, reverse=True)
    best = candidates[0]
    _log("best_ocr_title candidates:", candidates, "->", best)
    return best

def normalize_title(s: str) -> str:
    s = s.strip()
    trans = str.maketrans({
        "—":"-","–":"-","“":"\"","”":"\"","‘":"'","’":"'",
        "０":"0","１":"1","２":"2","３":"3","４":"4","５":"5",
        "６":"6","７":"7","８":"8","９":"9",
    })
    s = s.translate(trans).replace("|","I")
    s = "".join(ch for ch in s if (ch.isalnum() or ch.isspace() or ch in "-!'?"))
    return " ".join(s.split()).lower()

def compact(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum()).lower()

def find_event_by_title(global_idx: Dict[str, Dict[str, Any]],
                        ocr_title_str: str,
                        min_score: int = EVENT_FUZZY_SCORE):
    q = normalize_title(ocr_title_str)
    if not q: return None, None
    q_c = compact(q)
    best_key, best_score = None, -1
    best_compact = -1
    for key in global_idx.keys():
        s1 = fuzz.token_sort_ratio(q, key)
        s2 = fuzz.partial_ratio(q, key)
        s3 = fuzz.QRatio(q, key)
        s4 = fuzz.ratio(q_c, compact(key))
        score = max(s1, s2, s3, s4)
        if s4 > best_compact: best_compact = s4
        if score > best_score:
            best_key, best_score = key, score
    if best_key:
        compact_ok = fuzz.ratio(q_c, compact(best_key)) >= 90
        if best_score >= min_score or compact_ok:
            return global_idx[best_key], int(max(best_score, best_compact))
    return None, int(best_score)

# =========================
# Events index
# =========================
def _summarize_common_outcome(effects: Dict[str, Any]) -> List[str]:
    lines = []
    for key in ("energy","mood","skill_points","random_stat","last_trained_stat","all_stats",
                "maximum_energy","speed","stamina","power","guts","wisdom",
                "yayoi_akikawa_bond","etsuko_otonashi_bond"):
        if key in effects:
            val = effects[key]
            if isinstance(val, list) and len(val) == 2:
                lines.append(f"{key.replace('_',' ').title()} {val[0]}/{val[1]}")
            else:
                sign = "+" if isinstance(val,(int,float)) and val>=0 else ""
                pretty = key.replace("_"," ").title()
                lines.append(f"{pretty} {sign}{val}")
    if effects.get("status_heal_negative"):
        lines.append("Heal negative status effects")
    for s in effects.get("status_add", []):
        lines.append(f"Get {s} status")
    for s in effects.get("skills_gain", []):
        lines.append(f"Obtain {s} skill")
    if "skill_hint" in effects:
        lines.append("(random) Hint for a skill related to the race" if effects["skill_hint"]=="race_related"
                     else f"Hint: {effects['skill_hint']}")
    return lines

def normalize_event_from_card_file(card_name: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    title = raw.get("name","").strip()
    choices = []
    for label_key, human in (("top_option","Top Option"),
                             ("middle_option","Middle Option"),
                             ("bottom_option","Bottom Option")):
        if label_key in raw:
            arr = raw[label_key] if isinstance(raw[label_key], list) else [str(raw[label_key])]
            choices.append({"label": human, "lines": arr})
    return {"title": title, "origin": f"card::{card_name}", "choices": choices}

def normalize_event_from_legacy(actor: str, kind: str, ev: Dict[str, Any]) -> Dict[str, Any]:
    title = ev.get("title","").strip()
    choices = []
    for ch in ev.get("choices", []):
        label = ch.get("label","Option")
        eff   = ch.get("effects", {})
        lines = []
        if isinstance(eff, dict) and eff:
            for k,v in eff.items():
                if isinstance(v, list) and len(v)==2 and all(isinstance(x,(int,float)) for x in v):
                    lines.append(f"{k.replace('_',' ').title()} {v[0]}/{v[1]}")
                elif isinstance(v,(int,float)):
                    sign = "+" if v>=0 else ""
                    lines.append(f"{k.replace('_',' ').title()} {sign}{v}")
                else:
                    lines.append(f"{k}: {v}")
        else:
            val = ch.get("text") or ch.get("line") or ""
            if val: lines.append(str(val))
        if not lines: lines = ["(no effect data)"]
        choices.append({"label": label, "lines": lines})
    return {"title": title, "origin": f"{kind}::{actor}", "choices": choices}

def normalize_event_from_common(raw: Dict[str, Any]) -> Dict[str, Any]:
    title = raw.get("event_name","").strip()
    choices = []
    for opt in raw.get("options", []):
        label = opt.get("name","Option")
        lines = []
        for outcome in opt.get("outcomes", []):
            eff = outcome.get("effects", {})
            chance = outcome.get("chance", None)
            summary = _summarize_common_outcome(eff)
            if chance is not None:
                pct = int(round(chance*100)) if 0 < chance <= 1 else int(chance)
                if summary:
                    summary[0] = f"(~{pct}%) " + summary[0]
                else:
                    summary = [f"(~{pct}%)"]
            if summary: lines.append(" / ".join(summary))
        if not lines: lines = ["(no effect data)"]
        choices.append({"label": label, "lines": lines})
    return {"title": title, "origin": "common", "choices": choices}

def build_global_event_index() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    if COMMON_FILE.exists():
        try:
            common = json.load(open(COMMON_FILE, "r", encoding="utf-8"))
            for ev in common.get("common_events", []):
                norm = normalize_event_from_common(ev)
                key  = normalize_title(norm["title"])
                if key: index[key] = norm
        except Exception as e:
            if DEBUG_LOG: print("[load common] error:", e)
    if CARDS_DIR.exists():
        for jf in CARDS_DIR.glob("*.json"):
            if jf.name.lower() in ("events.json","common.json"): continue
            try:
                data = json.load(open(jf, "r", encoding="utf-8"))
                card_name = data.get("card", jf.stem)
                for ev in data.get("events", []):
                    norm = normalize_event_from_card_file(card_name, ev)
                    key  = normalize_title(norm["title"])
                    if key: index[key] = norm
            except Exception as e:
                if DEBUG_LOG: print("[load card file] error:", jf, e)
    for kind_dir, kind in ((CARDS_DIR, "card"), (TRAINEES_DIR, "trainee")):
        if not kind_dir.exists(): continue
        for actor_dir in kind_dir.iterdir():
            if not actor_dir.is_dir(): continue
            evfile = actor_dir / "events.json"
            if not evfile.exists(): continue
            try:
                events = json.load(open(evfile, "r", encoding="utf-8")).get("events", [])
            except Exception as e:
                if DEBUG_LOG: print("[load legacy] error:", evfile, e); continue
            actor = actor_dir.name
            for ev in events:
                norm = normalize_event_from_legacy(actor, kind, ev)
                key  = normalize_title(norm["title"])
                if key: index[key] = norm
    if DEBUG_LOG: print(f"[UmaHelper] Indexed {len(index)} titles")
    return index

# =========================
# ROI Editor
# =========================
class RoiOverlay(QWidget):
    saved = Signal(dict)
    EDGE_MARGIN = 8
    def __init__(self, screens_map: Dict[str, Dict[str, List[int]]], screen_id: str, roi_key: str, screen_geo: QRect):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.screens_map = {sid: {k:v[:] for k,v in areas.items()} for sid,areas in screens_map.items()}
        self.sid = screen_id
        self.geo = screen_geo
        self.move(self.geo.topLeft())
        self.resize(self.geo.size())
        if self.sid not in self.screens_map or not self.screens_map[self.sid]:
            self.screens_map[self.sid] = default_rois_for_screen(self.geo)
        self.key = roi_key if roi_key in self.screens_map[self.sid] else list(self.screens_map[self.sid].keys())[0]
        self.drag_mode = None
        self.resize_edge = None
        self.drag_start = QPoint()
        self.rect_start = None
        self._hover_save = False
        self._hover_cancel = False
        self.btn_save = QRect(20, 20, 90, 36)
        self.btn_cancel = QRect(120, 20, 110, 36)
        self.info_rect = QRect(20, 66, 800, 26)

    def _to_local(self, rect: List[int]) -> QRect:
        return QRect(rect[0]-self.geo.x(), rect[1]-self.geo.y(), rect[2]-rect[0], rect[3]-rect[1])
    def _to_abs(self, qr: QRect) -> List[int]:
        l = qr.x() + self.geo.x(); t = qr.y() + self.geo.y()
        r = l + qr.width(); b = t + qr.height()
        return _ensure_rect([l,t,r,b])

    def _edge_hit(self, pos: QPoint, qr: QRect):
        m = self.EDGE_MARGIN
        left   = abs(pos.x() - qr.left()) <= m
        right  = abs(pos.x() - qr.right()) <= m
        top    = abs(pos.y() - qr.top()) <= m
        bottom = abs(pos.y() - qr.bottom()) <= m
        if left and top: return ("l","t")
        if right and top: return ("r","t")
        if left and bottom: return ("l","b")
        if right and bottom: return ("r","b")
        if left: return ("l",)
        if right: return ("r",)
        if top: return ("t",)
        if bottom: return ("b",)
        return None

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0,0,0,120))
        p.fillRect(self.btn_save, QColor(33,150,243, 220) if self._hover_save else QColor(33,150,243, 190))
        p.fillRect(self.btn_cancel, QColor(244,67,54, 220) if self._hover_cancel else QColor(244,67,54, 190))
        p.setPen(Qt.white); p.setFont(QFont("Segoe UI", 11, QFont.Medium))
        p.drawText(self.btn_save, Qt.AlignCenter, "Save")
        p.drawText(self.btn_cancel, Qt.AlignCenter, "Cancel")
        p.setPen(QColor(255,255,255,220))
        p.setFont(QFont("Segoe UI", 10))
        msg = f"Editing {self.key} on {self.sid} | Drag to draw/move/resize."
        p.drawText(self.info_rect, Qt.AlignLeft | Qt.AlignVCenter, msg)
        for name, rect in self.screens_map[self.sid].items():
            qr = self._to_local(rect)
            if name == self.key:
                fill = QColor(0, 155, 255, 50); pen = QPen(QColor(0, 210, 255, 230), 2, Qt.DashLine)
            else:
                fill = QColor(255, 255, 255, 20); pen = QPen(QColor(255, 255, 255, 120), 1, Qt.DotLine)
            p.fillRect(qr, fill); p.setPen(pen); p.drawRect(qr)
            p.setPen(Qt.white); p.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
            p.drawText(qr.adjusted(4, -20, 0, 0), Qt.AlignLeft | Qt.AlignTop, name)

    def mouseMoveEvent(self, e: QMouseEvent):
        pt = e.position().toPoint()
        self._hover_save = self.btn_save.contains(pt)
        self._hover_cancel = self.btn_cancel.contains(pt)
        qr = self._to_local(self.screens_map[self.sid][self.key])
        if self.drag_mode == "new":
            x1, y1 = self.drag_start.x()-self.geo.x(), self.drag_start.y()-self.geo.y()
            x2, y2 = pt.x(), pt.y()
            l, r = sorted([x1, x2]); t, b = sorted([y1, y2])
            self.screens_map[self.sid][self.key] = self._to_abs(QRect(l, t, r - l, b - t)); self.update(); return
        if self.drag_mode == "move":
            delta = e.globalPosition().toPoint() - self.drag_start
            moved = QRect(self._to_local(self.screens_map[self.sid][self.key])); moved.moveTopLeft(moved.topLeft() + delta)
            moved.moveLeft(max(0, min(moved.left(), self.width()-moved.width())))
            moved.moveTop(max(0, min(moved.top(), self.height()-moved.height())))
            self.screens_map[self.sid][self.key] = self._to_abs(moved); self.update(); return
        if self.drag_mode == "resize":
            moved = QRect(self._to_local(self.screens_map[self.sid][self.key]))
            if "l" in self.resize_edge: moved.setLeft(pt.x())
            if "r" in self.resize_edge: moved.setRight(pt.x())
            if "t" in self.resize_edge: moved.setTop(pt.y())
            if "b" in self.resize_edge: moved.setBottom(pt.y())
            if moved.width() < 10: moved.setWidth(10)
            if moved.height() < 10: moved.setHeight(10)
            moved = moved.intersected(self.rect())
            self.screens_map[self.sid][self.key] = self._to_abs(moved); self.update(); return
        edge = self._edge_hit(pt, qr)
        if edge in (("l","t"), ("r","b")): self.setCursor(Qt.SizeFDiagCursor)
        elif edge in (("r","t"), ("l","b")): self.setCursor(Qt.SizeBDiagCursor)
        elif edge in (("l",), ("r",)): self.setCursor(Qt.SizeHorCursor)
        elif edge in (("t",), ("b",)): self.setCursor(Qt.SizeVerCursor)
        elif qr.contains(pt): self.setCursor(Qt.SizeAllCursor)
        else: self.setCursor(Qt.CrossCursor)
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        pt = e.position().toPoint()
        if self.btn_save.contains(pt):
            self.saved.emit(self.screens_map); self.close(); return
        if self.btn_cancel.contains(pt):
            self.close(); return
        qr = self._to_local(self.screens_map[self.sid][self.key])

        def edge_hit(pos, qr_):
            m = self.EDGE_MARGIN
            left   = abs(pos.x() - qr_.left()) <= m
            right  = abs(pos.x() - qr_.right()) <= m
            top    = abs(pos.y() - qr_.top()) <= m
            bottom = abs(pos.y() - qr_.bottom()) <= m
            if left and top: return ("l","t")
            if right and top: return ("r","t")
            if left and bottom: return ("l","b")
            if right and bottom: return ("r","b")
            if left: return ("l",)
            if right: return ("r",)
            if top: return ("t",)
            if bottom: return ("b",)
            return None

        edge = edge_hit(pt, qr)
        self.drag_start = e.globalPosition().toPoint()
        if edge is not None:
            self.drag_mode = "resize"; self.resize_edge = edge; self.rect_start = QRect(qr); return
        if qr.contains(pt):
            self.drag_mode = "move"; self.rect_start = QRect(qr); return
        self.drag_mode = "new"; self.rect_start = QRect(); self.update()

    def mouseReleaseEvent(self, e: QMouseEvent):
        self.drag_mode = None; self.resize_edge = None
        super().mouseReleaseEvent(e)

# =========================
# Result Window
# =========================
class ResultWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._dragging = False
        self._drag_offset = QPoint()

        outer = QVBoxLayout(self); outer.setContentsMargins(12,12,12,12)
        self.card = QFrame(self); self.card.setObjectName("card"); outer.addWidget(self.card)

        sh = QGraphicsDropShadowEffect(self); sh.setBlurRadius(24); sh.setOffset(0, 8); sh.setColor(QColor(0,0,0,170))
        self.card.setGraphicsEffect(sh)

        cardLay = QVBoxLayout(self.card); cardLay.setContentsMargins(0,0,0,0); cardLay.setSpacing(0)
        header = QFrame(self.card); header.setObjectName("header")
        hl = QHBoxLayout(header); hl.setContentsMargins(12,8,12,8); hl.setSpacing(8)
        self.headerLbl = QLabel("Result"); self.headerLbl.setObjectName("titlebar")
        hl.addWidget(self.headerLbl, 1); cardLay.addWidget(header)

        body = QFrame(self.card); body.setObjectName("body")
        bl = QVBoxLayout(body); bl.setContentsMargins(16,12,16,14); bl.setSpacing(6)
        self.titleLbl = QLabel("Waiting for scan…"); self.titleLbl.setTextFormat(Qt.RichText)
        self.metaLbl  = QLabel(""); self.metaLbl.setTextFormat(Qt.RichText); self.metaLbl.setObjectName("meta")
        self.linesLay = QVBoxLayout(); self.linesLay.setSpacing(4)
        bl.addWidget(self.titleLbl); bl.addWidget(self.metaLbl); bl.addSpacing(4); bl.addLayout(self.linesLay)
        cardLay.addWidget(body)

        self.setStyleSheet("""
            QFrame#card { background-color: rgba(18,25,38,230); border-radius: 14px; border: 1px solid rgba(255,255,255,40); }
            QFrame#header { border-top-left-radius: 14px; border-top-right-radius: 14px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 rgba(0,153,255,240), stop:1 rgba(0,230,255,240)); }
            QLabel#titlebar { color: white; font: 600 13px 'Segoe UI'; letter-spacing: 0.5px; }
            QLabel { color: white; font: 15px 'Segoe UI'; }
            QLabel#meta { color: #90a4ae; font: 12px 'Segoe UI'; }
            QLabel.option { color: #e3f2fd; font: 15px 'Segoe UI'; }
            QLabel.sub { color: #cfd8dc; margin-left: 16px; font: 13px 'Segoe UI'; }
        """)
        self.resize(680, 280); self.move(920, 220)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()
        else:
            super().mousePressEvent(e)
    def mouseMoveEvent(self, e: QMouseEvent):
        if self._dragging and (e.buttons() & Qt.LeftButton):
            self.move(e.globalPosition().toPoint() - self._drag_offset); e.accept()
        else:
            super().mouseMoveEvent(e)
    def mouseReleaseEvent(self, e: QMouseEvent):
        self._dragging = False; super().mouseReleaseEvent(e)

    def _clear_lines(self):
        for i in reversed(range(self.linesLay.count())):
            w = self.linesLay.itemAt(i).widget()
            if w: w.setParent(None); w.deleteLater()

    def show_scanning(self):
        self.headerLbl.setText("Scanning…")
        self.titleLbl.setText("<i>Scanning…</i>")
        self.metaLbl.setText("")
        self._clear_lines()

    def show_result(self, ev: Optional[Dict[str, Any]], score: Optional[int], ocr_text: str):
        if not ev:
            self.headerLbl.setText("Result")
            self.titleLbl.setText("No match")
            self.metaLbl.setText(f"<span style='color:#90a4ae'>OCR: {ocr_text or '(… )'}</span>")
            self._clear_lines()
            return
        self.headerLbl.setText("Event Found")
        color = "#b2ff59" if score and score >= 93 else ("#ffd54f" if score and score >= 88 else "#ef9a9a")
        self.titleLbl.setText(f"<span style='color:white'>{ev['title']}</span>")
        self.metaLbl.setText(f"<span style='color:{color}'>Match score {score}</span> "
                             f"<span style='color:#90a4ae'>[{ev.get('origin','?')}]</span>")
        self._clear_lines()
        lines = self._format_choices(ev)
        for ln in lines:
            cls = "sub" if ln.startswith("    ") else "option"
            lab = QLabel(ln.strip() if cls == "sub" else ln)
            lab.setObjectName(cls); lab.setProperty("class", cls)
            lab.setTextFormat(Qt.RichText); lab.setWordWrap(True)
            self.linesLay.addWidget(lab)

    def _format_choices(self, ev: Dict[str, Any]) -> List[str]:
        out: List[str] = ["<span style='color:#b0bec5'>— Choices —</span>"]
        for chx in ev.get("choices", []):
            label = chx.get("label", "Option"); parts = chx.get("lines", [])
            if not parts: out.append(f"<b>{label}</b>: <i>(no data)</i>")
            else:
                out.append(f"<b>{label}</b>: {parts[0]}")
                for more in parts[1:2]:
                    out.append("    " + more)
        return out

# =========================
# Control Window (header-only dragging)
# =========================
class ControlWindow(QWidget):
    requestEditROI = Signal(str, str)   # screen_id, roi_key
    requestScan    = Signal(str)        # screen_id

    def __init__(self, screens_map: Dict[str, Dict[str, List[int]]]):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._dragging = False
        self._drag_offset = QPoint()

        self.screens_map = screens_map
        self.screens = list_screens()

        outer = QVBoxLayout(self); outer.setContentsMargins(12,12,12,12)
        self.card = QFrame(self); self.card.setObjectName("card"); outer.addWidget(self.card)

        sh = QGraphicsDropShadowEffect(self); sh.setBlurRadius(24); sh.setOffset(0,8); sh.setColor(QColor(0,0,0,170))
        self.card.setGraphicsEffect(sh)

        cardLay = QVBoxLayout(self.card); cardLay.setContentsMargins(0,0,0,0); cardLay.setSpacing(0)
        header = QFrame(self.card); header.setObjectName("header")
        hl = QHBoxLayout(header); hl.setContentsMargins(12,8,12,8); hl.setSpacing(8)
        self.titleLbl = QLabel("UmaHelper — Control"); self.titleLbl.setObjectName("titlebar")
        hl.addWidget(self.titleLbl, 1); cardLay.addWidget(header)

        body = QFrame(self.card); body.setObjectName("body")
        bl = QVBoxLayout(body); bl.setContentsMargins(16,12,16,14); bl.setSpacing(10)

        row1 = QHBoxLayout()
        self.cboScreen = QComboBox()
        for sid, _s, geo, label in self.screens:
            self.cboScreen.addItem(label, sid)
        row1.addWidget(QLabel("Screen:")); row1.addWidget(self.cboScreen, 1)

        row2 = QHBoxLayout()
        self.cboRoi = QComboBox(); self._refresh_roi_list_for_current_screen()
        row2.addWidget(QLabel("ROI:")); row2.addWidget(self.cboRoi, 1)

        row3 = QHBoxLayout()
        self.btnEdit = QPushButton("Edit ROI")
        self.btnScan = QPushButton("Scan")
        row3.addWidget(self.btnEdit); row3.addWidget(self.btnScan)

        for b in (self.btnEdit, self.btnScan): b.setCursor(Qt.PointingHandCursor)

        bl.addLayout(row1); bl.addLayout(row2); bl.addLayout(row3)
        cardLay.addWidget(body)

        self.setStyleSheet("""
            QFrame#card { background-color: rgba(18,25,38,230); border-radius: 14px; border: 1px solid rgba(255,255,255,40); }
            QFrame#header { border-top-left-radius: 14px; border-top-right-radius: 14px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 rgba(0,153,255,240), stop:1 rgba(0,230,255,240)); }
            QLabel#titlebar { color: white; font: 600 13px 'Segoe UI'; letter-spacing: 0.5px; }
            QFrame#body { background-color: transparent; }
            QLabel { color: white; font: 14px 'Segoe UI'; }
            QComboBox { background-color: rgba(0,0,0,40); color: white; border: 1px solid rgba(255,255,255,50);
                        padding: 4px 8px; border-radius: 6px; font: 12px 'Segoe UI'; }
            QPushButton { background-color: rgba(255,255,255,30); color: white; border: 1px solid rgba(255,255,255,50);
                          padding: 6px 10px; border-radius: 8px; font: 600 12px 'Segoe UI'; }
            QPushButton:hover { background-color: rgba(255,255,255,40); }
        """)

        # Header-only dragging
        self.header = header
        self.header.setMouseTracking(True)
        self.header.installEventFilter(self)
        for child in self.header.findChildren(QWidget):
            child.installEventFilter(self)

        self.cboScreen.setCurrentIndex(0)
        self._ensure_screen_defaults()
        self._refresh_roi_list_for_current_screen()
        _log("ControlWindow ready; current screen:", self.cboScreen.currentData())

        self.resize(460, 200); self.move(200, 220)

        # hooks
        self.cboScreen.currentIndexChanged.connect(self._on_screen_changed)
        self.btnEdit.clicked.connect(self._emit_edit)
        self.btnScan.clicked.connect(self._emit_scan)

    def eventFilter(self, obj, e):
        if obj is self.header or obj in self.header.findChildren(QWidget):
            if e.type() == QEvent.MouseButtonPress and e.button() == Qt.LeftButton:
                self._dragging = True
                self._drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
                return True
            if e.type() == QEvent.MouseMove and self._dragging and (e.buttons() & Qt.LeftButton):
                self.move(e.globalPosition().toPoint() - self._drag_offset)
                return True
            if e.type() == QEvent.MouseButtonRelease:
                self._dragging = False
                return True
        return super().eventFilter(obj, e)

    def current_screen_id(self) -> str:
        sid = self.cboScreen.currentData()
        return "0" if sid is None else str(sid)

    def _on_screen_changed(self, _i):
        self._ensure_screen_defaults()
        self._refresh_roi_list_for_current_screen()

    def _ensure_screen_defaults(self):
        sid = self.current_screen_id()
        geo = None
        for sid2, _s, g, _label in self.screens:
            if sid2 == sid: geo = g; break
        if sid not in self.screens_map or not self.screens_map[sid]:
            self.screens_map[sid] = default_rois_for_screen(geo)

    def _refresh_roi_list_for_current_screen(self):
        sid = self.current_screen_id()
        self.cboRoi.clear()
        if sid in self.screens_map and self.screens_map[sid]:
            self.cboRoi.addItems(list(self.screens_map[sid].keys()))
        else:
            self.cboRoi.addItems(list(DEFAULT_ROI_LOCAL.keys()))

    def _emit_edit(self):
        sid = self.current_screen_id()
        key = self.cboRoi.currentText()
        self.requestEditROI.emit(sid, key)

    def _emit_scan(self):
        sid = self.current_screen_id()
        _log("Scan clicked; emitting screen_id:", sid)
        self.requestScan.emit(sid)

# =========================
# Scan worker (runs off the UI thread)
# =========================
class ScanWorker(QObject):
    finished = Signal(object, object, str)  # ev (dict or None), score (int or None), ocr_text (str)
    failed   = Signal(str)

    def __init__(self, pil_image: Image.Image, index: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.pil_image = pil_image
        self.index = index

    def run(self):
        try:
            ocr = best_ocr_title(self.pil_image)
            ev, score = find_event_by_title(self.index, ocr, min_score=EVENT_FUZZY_SCORE)
            self.finished.emit(ev, score, ocr or "")
        except Exception as e:
            self.failed.emit(str(e))

# =========================
# App Controller (QObject so slots run on GUI thread)
# =========================
class UmaHelperApp(QObject):
    def __init__(self):
        super().__init__()
        self.index = build_global_event_index()
        self.screens_map = load_all_rois()

        self.control = ControlWindow(self.screens_map)
        self.result  = ResultWindow()
        self.roi_overlay = None

        self.control.requestEditROI.connect(self.open_roi_editor)
        self.control.requestScan.connect(self.scan_once)

        self._scan_thread: Optional[QThread] = None
        self._scan_worker: Optional[ScanWorker] = None

        self.control.show()
        self.result.show()

    def open_roi_editor(self, screen_id: str, roi_key: str):
        geo = None
        for sid, _s, g, _label in list_screens():
            if sid == screen_id: geo = g; break
        if geo is None: return
        if self.roi_overlay is not None:
            try: self.roi_overlay.close()
            except Exception: pass
        self.roi_overlay = RoiOverlay(self.screens_map, screen_id, roi_key, geo)
        self.roi_overlay.saved.connect(self._on_rois_saved)
        self.roi_overlay.show(); self.roi_overlay.raise_(); self.roi_overlay.activateWindow()

    def _on_rois_saved(self, new_map: Dict[str, Dict[str, List[int]]]):
        self.screens_map = new_map
        save_all_rois(self.screens_map)
        self.control.screens_map = self.screens_map
        self.control._refresh_roi_list_for_current_screen()

    def scan_once(self, screen_id: str):
        try:
            _log("scan_once start; screen_id:", screen_id)
            areas = self.screens_map.get(screen_id)
            if not areas:
                _log("No ROIs for screen; seeding defaults")
                geo = None
                for sid, _s, g, _label in list_screens():
                    if sid == screen_id: geo = g; break
                areas = default_rois_for_screen(geo)
                self.screens_map[screen_id] = areas
                save_all_rois(self.screens_map)

            title_rect = areas.get("title_text")
            _log("title_rect:", title_rect)
            if not title_rect:
                self.result.show_result(None, None, "(no title_text ROI)")
                return

            img = grab_abs_rect(title_rect)
            if not img:
                _log("MSS capture failed for rect:", title_rect)
                self.result.show_result(None, None, "(capture failed)")
                return

            # UI feedback + disable button while scanning
            self.result.show_scanning()
            self.control.btnScan.setEnabled(False)

            # Background thread for OCR/matching
            self._scan_thread = QThread()
            self._scan_worker = ScanWorker(img, self.index)
            self._scan_worker.moveToThread(self._scan_thread)

            self._scan_thread.started.connect(self._scan_worker.run)
            # Force queued connections so UI updates occur on GUI thread
            self._scan_worker.finished.connect(self._on_scan_finished, Qt.QueuedConnection)
            self._scan_worker.failed.connect(self._on_scan_failed, Qt.QueuedConnection)

            # Cleanup
            self._scan_worker.finished.connect(self._scan_thread.quit)
            self._scan_worker.finished.connect(self._scan_worker.deleteLater)
            self._scan_thread.finished.connect(self._scan_thread.deleteLater)
            self._scan_worker.failed.connect(self._scan_thread.quit)
            self._scan_worker.failed.connect(self._scan_worker.deleteLater)

            self._scan_thread.start()

        except Exception as e:
            import traceback
            _log("Exception in scan_once:", e); traceback.print_exc()
            self.result.show_result(None, None, f"(error: {e})")
            self.control.btnScan.setEnabled(True)

    @Slot(object, object, str)
    def _on_scan_finished(self, ev, score, ocr_text: str):
        self.control.btnScan.setEnabled(True)
        self.result.show_result(ev, score, ocr_text)

    @Slot(str)
    def _on_scan_failed(self, err: str):
        self.control.btnScan.setEnabled(True)
        self.result.show_result(None, None, f"(error: {err})")

# =========================
# Entry
# =========================
def main():
    enable_dpi_awareness()
    app = QApplication(sys.argv)
    helper = UmaHelperApp()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
