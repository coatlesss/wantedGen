"""
Wanted Poster Maker (Folder drop + Flask gallery)

Install:
  pip install flask pillow

Run:
  python wantedGen.py

Use:
- Put your template image path in WANTED_TEMPLATE_PATH.
- Drop face images into ./input_faces
  Example: John_Doe.jpg -> name becomes "JOHN DOE"
- The app auto-scans every few seconds and generates posters.
- Open: http://127.0.0.1:5000
"""

from __future__ import annotations

import time
import re
from pathlib import Path

from flask import Flask, send_from_directory, render_template_string
from PIL import Image, ImageDraw, ImageFont, ImageOps

# =========================
# CONFIG
# =========================
# IMPORTANT: Change this to your real template path on Windows.
# Example:
# WANTED_TEMPLATE_PATH = r"C:\Users\bedfa\Desktop\Hackathon\template.png"
WANTED_TEMPLATE_PATH = r"C:\Users\bedfa\wantedGen\template.jpg"

INPUT_DIR = Path("input_faces")
OUTPUT_DIR = Path("static/generated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
FACE_SIZE = (240, 240)

# Tuned for the provided template (adjust if needed)
PHOTO_BOX = (148, 393, 1081, 1276)          # where the face goes (left, top, right, bottom)
NAME_PLATE_BOX = (336, 1381, 878, 1454)    # where the name text goes

SCAN_INTERVAL_SECONDS = 2


# =========================
# HELPERS
# =========================
def allowed_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTS


def secure_display_name(filename: str) -> str:
    """
    'John_Doe.png' -> 'John Doe'
    """
    stem = Path(filename).stem
    stem = stem.replace("_", " ").strip()
    stem = re.sub(r"[^A-Za-z0-9 \-']", "", stem).strip()
    return stem or "UNKNOWN"


def load_font(size: int):
    candidates = [
        "DejaVuSans-Bold.ttf",
        "DejaVuSerif-Bold.ttf",
        "arialbd.ttf",          # Arial Bold (common on Windows)
        "Arial Bold.ttf",
        "timesbd.ttf",          # Times Bold (sometimes)
        "Times New Roman Bold.ttf",
        "Arial.ttf",
        "Times New Roman.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def fit_text(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int]) -> ImageFont.ImageFont:
    left, top, right, bottom = box
    max_w = right - left
    max_h = bottom - top

    for size in range(120, 12, -2):
        font = load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= max_w and h <= max_h:
            return font
    return load_font(18)


from PIL import Image, ImageDraw, ImageFont, ImageOps

DEBUG_BOXES = False  # set to False when done tuning

def make_wanted_poster(face_img: Image.Image, person_name: str) -> Image.Image:
    template = Image.open(WANTED_TEMPLATE_PATH).convert("RGBA")
    W, H = template.size

    # --- Use ratio-based boxes so it works even if template size changes ---
    # These ratios are tuned for your exact template look.
    # If they're still slightly off, use DEBUG_BOXES to adjust.
    PHOTO_BOX = (
        int(W * 0.115),  # left
        int(H * 0.205),  # top
        int(W * 0.885),  # right
        int(H * 0.735),  # bottom
    )
    NAME_PLATE_BOX = (
        int(W * 0.185),  # left
        int(H * 0.765),  # top
        int(W * 0.815),  # right
        int(H * 0.845),  # bottom
    )

    # 1) Normalize face to 240x240 (your requirement)
    face_240 = ImageOps.fit(face_img.convert("RGBA"), (240, 240), method=Image.Resampling.LANCZOS)

    # Optional vintage tone (comment out if you want original colors)
    face_240 = ImageOps.colorize(
        ImageOps.grayscale(face_240), black="#2b1a0f", white="#f6e3b2"
    ).convert("RGBA")

    # 2) IMPORTANT: scale that 240x240 up to fill the big PHOTO_BOX (so it isn't tiny)
    l, t, r, b = PHOTO_BOX
    box_w, box_h = (r - l), (b - t)

    # Keep it "photo-like": fit to the box (crop to fill) so it fills the frame.
    face_for_poster = ImageOps.fit(face_240, (box_w, box_h), method=Image.Resampling.LANCZOS)

    out = template.copy()
    out.alpha_composite(face_for_poster, (l, t))

    # 3) Draw the name centered in NAME_PLATE_BOX
    draw = ImageDraw.Draw(out)
    text = person_name.upper()

    font = fit_text(draw, text, NAME_PLATE_BOX)

    nl, nt, nr, nb = NAME_PLATE_BOX
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    tx = nl + (nr - nl - tw) // 2
    ty = nt + (nb - nt - th) // 2

    ink = (72, 33, 20, 255)
    draw.text((tx + 2, ty + 2), text, font=font, fill=(0, 0, 0, 120))
    draw.text((tx, ty), text, font=font, fill=ink)

    # 4) Debug overlay boxes so you can see what the script thinks the regions are
    if DEBUG_BOXES:
        dbg = ImageDraw.Draw(out)
        dbg.rectangle(PHOTO_BOX, outline=(255, 0, 0, 255), width=6)      # red photo box
        dbg.rectangle(NAME_PLATE_BOX, outline=(0, 255, 0, 255), width=6) # green name box

    return out.convert("RGB")

def output_filename_for(face_path: Path) -> str:
    # Output uses the same stem, but ends in .jpg
    safe_stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", face_path.stem).strip("_") or "UNKNOWN"
    return f"{safe_stem}.jpg"


def process_new_files() -> int:
    """
    Scan INPUT_DIR and generate posters for any faces that don't have outputs yet.
    Returns how many posters were created this scan.
    """
    created = 0
    for face_path in INPUT_DIR.iterdir():
        if not face_path.is_file():
            continue
        if not allowed_file(face_path):
            continue

        out_name = output_filename_for(face_path)
        out_path = OUTPUT_DIR / out_name
        if out_path.exists():
            continue  # already processed

        try:
            with Image.open(face_path) as face_img:
                name = secure_display_name(face_path.name)
                poster = make_wanted_poster(face_img, name)
                poster.save(out_path, quality=92, optimize=True)
                created += 1
        except Exception as e:
            print(f"[WARN] Failed to process {face_path.name}: {e}")

    return created


# =========================
# FLASK (VIEW ONLY)
# =========================
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Wanted Posters</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    .wrap { max-width: 980px; margin: 0 auto; }
    .muted { color: #666; font-size: 14px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 12px; }
    img { width: 100%; height: auto; border-radius: 10px; border: 1px solid #eee; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
<div class="wrap">
  <h1>Wanted Posters</h1>
  <p class="muted">
    Drop face images into <code>{{ input_dir }}</code>. This page is just for viewing.
    Auto-scan runs every {{ interval }} seconds.
  </p>

  {% if images %}
    <div class="grid">
      {% for img in images %}
        <a href="{{ url_for('generated_file', filename=img) }}">
          <img src="{{ url_for('generated_file', filename=img) }}" alt="{{ img }}">
        </a>
      {% endfor %}
    </div>
  {% else %}
    <p class="muted">No posters yet. Add a face image to <code>{{ input_dir }}</code>.</p>
  {% endif %}
</div>
</body>
</html>
"""

_last_scan = 0.0

@app.before_request
def auto_scan_folder():
    global _last_scan
    now = time.time()
    if now - _last_scan >= SCAN_INTERVAL_SECONDS:
        made = process_new_files()
        if made:
            print(f"[INFO] Created {made} poster(s).")
        _last_scan = now


@app.get("/")
def index():
    images = sorted([p.name for p in OUTPUT_DIR.glob("*.jpg")], reverse=True)
    return render_template_string(
        INDEX_HTML,
        images=images,
        input_dir=str(INPUT_DIR),
        interval=SCAN_INTERVAL_SECONDS,
    )


@app.get("/generated/<path:filename>")
def generated_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    template_path = Path(WANTED_TEMPLATE_PATH)
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found at: {WANTED_TEMPLATE_PATH}\n"
            "Fix WANTED_TEMPLATE_PATH to point to your wanted poster image on your PC."
        )

    print(f"[INFO] Watching folder: {INPUT_DIR.resolve()}")
    print(f"[INFO] Output folder:   {OUTPUT_DIR.resolve()}")
    print("[INFO] Open: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)