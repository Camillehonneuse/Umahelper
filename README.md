# UmaHelper

**UmaHelper** is a desktop utility for **Uma Musume** players that can automatically recognize event titles on-screen and instantly show the correct event choices and effects.  
It supports **multi-monitor setups**, **split-window ROI overlays**, threaded OCR scanning with **PaddleOCR**, and a modern, floating **PySide6** interface.

With this tool, you can hover the ROI over the game window, scan the event title, and immediately see the right answer without switching windows or alt-tabbing.

---

## ✨ Features

- **Multi-monitor support** – choose which display to scan from.
- **Custom ROI editor** – draw, move, and resize capture areas with a visual overlay.
- **Real-time OCR** – powered by **PaddleOCR** (fast, accurate, supports mixed-case & punctuation).
- **Threaded scanning** – keeps the UI responsive while processing.
- **Data-driven matching** – fuzzy-searches across JSON databases of:
  - Card events
  - Trainee events
  - Common events
- **Modern floating UI** – draggable, shadowed cards with clear typography.
- **Persistent settings** – saves ROIs per monitor in `roi/capture_areas.json`.
- **Packaged build** – distributed as a single `.exe` with **PyInstaller** (no Python install required).

---

## 📦 Installation

### From Source (Developer Mode)
1. Install Python 3.9+ (Windows recommended).
2. Install dependencies:
   ```bash
   pip install mss pillow paddleocr rapidfuzz PySide6 numpy opencv-python
   ```
3. Run the app:
   ```bash
   python umahelper.py
   ```

### From Release (Executable)
1. Download the latest `UmaHelper.exe` from the [Releases](./releases) page.
2. Run `UmaHelper.exe` — no installation needed.

---

## 📂 Project Structure

```
UmaHelper/
├── main.py             # Main application
├── assets/
│   ├── cards/               # Card event data (JSON files)
│   ├── trainees/            # Trainee event data (JSON files)
│   └── common/events.json   # Common events database
├── roi/
│   └── capture_areas.json   # Saved ROI coordinates (auto-created)
└── README.md                # This file
```

---

## 🚀 Usage

1. **Launch UmaHelper**  
   - If running from source, use:  
     ```bash
     python main.py
     ```
   - If using `.exe`, just double-click it.

2. **Control Window**  
   - Select your monitor from the dropdown.
   - Select which ROI to edit (e.g., `title_text`).
   - Click **Edit ROI** to visually adjust the capture area.
   - Click **Scan** to OCR and match the current on-screen event.

3. **Result Window**  
   - Shows event name, match score, and origin (card/trainee/common).
   - Lists available choices and their effects.
   - You can move this window anywhere on your screen.

---

## 🖱 ROI Editor

- **Draw a new ROI** – click & drag on empty space.
- **Move an ROI** – drag inside the highlighted area.
- **Resize an ROI** – drag edges or corners.
- **Save** – stores your ROIs in `roi/capture_areas.json`.
- **Cancel** – discards your changes.

Each monitor can have **its own** ROI settings.

---

## ⚙️ Configuration

You can adjust these constants at the top of `umahelper.py`:

| Setting             | Description |
|---------------------|-------------|
| `EVENT_FUZZY_SCORE` | Minimum match score (default `80`). Lower for looser matching. |
| `DEBUG_LOG`         | Set `True` to print debug logs to the console. |

---

## 🧩 Data Format

(This part is here if you want to reuse this for another completely different project, or if you're nice enough to help me complete the "Asset" part  ^^)


### Trainee file Format
```json
{
  "actor_id": "agnes_tachyon",
  "actor_type": "card",
  "language": "en",
  "events": [
    {
      "event_id": "ev.at_tachyons_pace",
      "title": "At Tachyon's Pace",
      "choices": [
        { "index": 1, "label": "Top Option", "effects": { "guts": 10 } }
      ]
    }
  ]
}
```

### Card File Format
```json
{
  "card": "Kitasan Black",
  "type": "Speed - SSR",
  "events": [
    {
      "name": "Paying It Forward",
      "top_option": ["Energy +10", "Mood +1", "Kitasan Black bond +5"]
    }
  ]
}
```

Event files live in:
- `assets/cards/`
- `assets/trainees/`
- `assets/common/events.json`

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Credits

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) – OCR engine.
- [PySide6](https://doc.qt.io/qtforpython/) – GUI framework.
- [mss](https://github.com/BoboTiG/python-mss) – Cross-platform screen capture.
- [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) – Fuzzy string matching.
- [OpenCV](https://opencv.org/) – Image preprocessing.

## IDK

Hey ! It's me, thanks for reading, this project is still in development, and I'm not that good with python, so don't have too much hope in this project.
