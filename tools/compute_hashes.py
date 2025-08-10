from PIL import Image
import imagehash, json, pathlib

base = pathlib.Path("assets")
for folder in ["cards", "trainees"]:
    p = base / folder
    if not p.exists(): 
        continue
    for actor_dir in p.iterdir():
        portrait = actor_dir / "portrait.png"
        if portrait.exists():
            h = imagehash.phash(Image.open(portrait).convert("RGB"))
            (actor_dir / "hash.json").write_text(json.dumps({"phash": str(h)}, indent=2), encoding="utf-8")
            print("Hashed:", portrait, "→", h)
