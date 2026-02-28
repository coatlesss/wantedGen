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

import json
import time
import re
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory, render_template_string, stream_with_context
from PIL import Image, ImageDraw, ImageFont, ImageOps

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
WANTED_TEMPLATE_PATH = BASE_DIR / "template.jpg"

INPUT_DIR = BASE_DIR / "input_faces"
OUTPUT_DIR = BASE_DIR / "static" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
FACE_SIZE = (240, 240)

# Tuned for the provided template (adjust if needed)
PHOTO_BOX = (148, 393, 1081, 1276)          # where the face goes (left, top, right, bottom)
NAME_PLATE_BOX = (336, 1381, 878, 1454)    # where the name text goes

SCAN_INTERVAL_SECONDS = 3
STREAM_HEARTBEAT_SECONDS = 1


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


def list_generated_images() -> list[dict[str, str | int]]:
    posters = []
    for image_path in sorted(OUTPUT_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True):
        posters.append(
            {
                "name": image_path.name,
                "label": secure_display_name(image_path.name),
                "url": f"/generated/{image_path.name}",
                "modified": int(image_path.stat().st_mtime),
            }
        )
    return posters


def stream_poster_updates():
    last_payload = None
    last_scan = 0.0

    while True:
        now = time.time()
        if now - last_scan >= SCAN_INTERVAL_SECONDS:
            made = process_new_files()
            if made:
                print(f"[INFO] Created {made} poster(s).")
            last_scan = now

        payload = json.dumps({"images": list_generated_images()})
        if payload != last_payload:
            yield f"data: {payload}\n\n"
            last_payload = payload
        else:
            yield ": keepalive\n\n"

        time.sleep(STREAM_HEARTBEAT_SECONDS)


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
        if out_path.exists() and out_path.stat().st_mtime >= face_path.stat().st_mtime:
            continue  # already processed and up to date

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
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Wanted Posters</title>
  <style>
    :root{
      --night:#120a06;
      --sand1:#2a160d;
      --sand2:#3a210f;
      --paper:#f3e2c6;
      --paper2:#ecd2a2;
      --ink:#2b1a10;
      --ink2:#4a2a18;
      --shadow: 0 18px 55px rgba(0,0,0,.45);
      --radius: 18px;
    }
    *{ box-sizing:border-box; }
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      color: var(--paper);
      background:
        radial-gradient(1200px 600px at 15% -10%, rgba(255,180,90,.18), transparent 65%),
        radial-gradient(900px 500px at 85% 0%, rgba(90,180,255,.10), transparent 60%),
        linear-gradient(180deg, var(--night) 0%, var(--sand1) 55%, var(--sand2) 100%);
      min-height:100vh;
      overflow-x:hidden;
    }

    .haze{
      position:fixed; inset:-100px;
      background: radial-gradient(900px 500px at 50% 40%, rgba(255,255,255,.08), transparent 60%);
      pointer-events:none;
      filter: blur(2px);
      opacity:.55;
      z-index:0;
    }

    /* cactus png decorations */
    .cactus{
      position: fixed;
      bottom: -10px;
      width: 260px;
      opacity: .26;
      pointer-events:none;
      z-index:0;
      filter: drop-shadow(0 10px 25px rgba(0,0,0,.45));
    }
    .cactus.left{ left: 10px; }
    .cactus.right{ right: 10px; transform: scaleX(-1); }

    .wrap{
      position:relative;
      z-index:1;
      max-width: 1120px;
      margin: 0 auto;
      padding: 26px;
    }

    .topbar{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:16px;
      padding: 18px 18px;
      border-radius: var(--radius);
      background: linear-gradient(180deg, rgba(243,226,198,.12), rgba(243,226,198,.06));
      border: 1px solid rgba(243,226,198,.18);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    .brand{ display:flex; flex-direction:column; gap:4px; }
    .title{
      margin:0;
      letter-spacing:.08em;
      text-transform:uppercase;
      font-weight:900;
      font-size: 18px;
    }
    .sub{
      margin:0;
      color: rgba(243,226,198,.75);
      font-size: 13px;
      line-height: 1.35;
    }

    .badge{
      display:inline-flex;
      align-items:center;
      gap:10px;
      padding:10px 12px;
      border-radius: 999px;
      border: 1px solid rgba(243,226,198,.20);
      background: rgba(0,0,0,.18);
      color: rgba(243,226,198,.85);
      font-size: 12px;
      white-space: nowrap;
    }
    .dot{
      width:8px; height:8px; border-radius:999px;
      background: #f4c67a;
      box-shadow: 0 0 0 6px rgba(244,198,122,.15);
      animation: pulse 1.8s ease-in-out infinite;
    }

    @keyframes pulse{
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.18); opacity: .7; }
    }

    .paper{
      margin-top: 18px;
      border-radius: calc(var(--radius) + 6px);
      background:
        radial-gradient(900px 600px at 20% 10%, rgba(255,255,255,.25), transparent 55%),
        radial-gradient(700px 500px at 85% 25%, rgba(0,0,0,.05), transparent 60%),
        linear-gradient(180deg, var(--paper) 0%, var(--paper2) 100%);
      border: 1px solid rgba(43,26,16,.25);
      box-shadow: 0 25px 70px rgba(0,0,0,.35);
      padding: 18px;
      color: var(--ink);
      position: relative;
      overflow:hidden;
    }
    .paper:before{
      content:"";
      position:absolute; inset:0;
      background:
        radial-gradient(circle at 20% 30%, rgba(0,0,0,.05) 0 2px, transparent 3px),
        radial-gradient(circle at 70% 40%, rgba(0,0,0,.04) 0 1.6px, transparent 3px),
        radial-gradient(circle at 40% 80%, rgba(0,0,0,.04) 0 1.8px, transparent 3px);
      opacity:.45;
      pointer-events:none;
      mix-blend-mode:multiply;
    }

    .paper h2{
      margin:0 0 12px;
      font-size: 16px;
      letter-spacing: .06em;
      text-transform: uppercase;
      font-weight: 900;
      color: var(--ink2);
      position:relative;
      z-index:1;
    }

    .paper-head{
      position: relative;
      z-index: 1;
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:12px;
      margin-bottom: 12px;
    }

    .queue{
      font-size: 12px;
      font-weight: 800;
      color: rgba(43,26,16,.7);
      text-align: right;
    }

    .grid{
      position:relative;
      display:grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 14px;
      z-index:1;
    }

    .card{
      border-radius: 16px;
      overflow:hidden;
      border: 1px solid rgba(43,26,16,.22);
      background: rgba(255,255,255,.35);
      transition: transform .15s ease, box-shadow .15s ease;
      box-shadow: 0 8px 22px rgba(0,0,0,.12);
    }
    .card:hover{
      transform: translateY(-2px);
      box-shadow: 0 16px 34px rgba(0,0,0,.18);
    }

    .thumb{
      width:100%;
      aspect-ratio: 3/4;
      object-fit: cover;
      display:block;
      background: rgba(255,255,255,.35);
    }

    .meta{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:10px;
      padding: 10px 12px 12px;
      position:relative;
      z-index:1;
    }

    .name{
      font-size: 12px;
      font-weight: 900;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: var(--ink2);
      overflow:hidden;
      white-space:nowrap;
      text-overflow: ellipsis;
    }

    .btn{
      text-decoration:none;
      font-size:12px;
      font-weight:900;
      letter-spacing:.06em;
      text-transform: uppercase;
      color: var(--ink);
      padding: 8px 10px;
      border-radius: 12px;
      border: 1px solid rgba(43,26,16,.25);
      background: rgba(255,255,255,.55);
      transition: background .15s ease;
      white-space:nowrap;
    }
    .btn:hover{ background: rgba(255,255,255,.75); }

    .empty{
      padding: 16px;
      border-radius: 14px;
      border: 1px dashed rgba(43,26,16,.35);
      background: rgba(255,255,255,.25);
      color: rgba(43,26,16,.75);
      position:relative;
      z-index:1;
    }

    .hidden{ display:none; }

    code{
      background: rgba(0,0,0,.08);
      border: 1px solid rgba(0,0,0,.10);
      padding: 2px 6px;
      border-radius: 10px;
      color: var(--ink);
      font-weight: 800;
    }
  </style>
</head>
<body>
  <div class="haze"></div>

  <!-- cactus pngs (put them in static/assets/) -->
  <img class="cactus left"  src="/static/assets/cactus.webp"  alt="cactus left">
  <img class="cactus right" src="/static/assets/cactus.webp" alt="cactus right">

  <div class="wrap">
    <div class="topbar">
      <div class="brand">
        <h1 class="title">Cowboy Criminal Database</h1>
      </div>
      <div class="badge"><span class="dot"></span> <span id="status-text">Connecting to live upload stream</span></div>
    </div>

    <div class="paper">
      <div class="paper-head">
        <h2>Most Wanted</h2>
        <div class="queue">Upload target: <code>{{ input_dir }}</code></div>
      </div>

      <div id="gallery" class="grid{% if not images %} hidden{% endif %}">
        {% for img in images %}
          <div class="card">
            <a href="{{ img.url }}">
              <img class="thumb" src="{{ img.url }}" alt="{{ img.name }}">
            </a>
            <div class="meta">
              <div class="name">{{ img.label }}</div>
              <a class="btn" href="{{ img.url }}">Open</a>
            </div>
          </div>
        {% endfor %}
      </div>

      <div id="empty-state" class="empty{% if images %} hidden{% endif %}">
        No posters yet. The site is waiting for the Jetson to add a face image to <code>{{ input_dir }}</code>.
      </div>
    </div>
  </div>

  <script>
    const gallery = document.getElementById("gallery");
    const emptyState = document.getElementById("empty-state");
    const statusText = document.getElementById("status-text");
    function renderCards(images) {
      if (!images.length) {
        gallery.classList.add("hidden");
        emptyState.classList.remove("hidden");
        statusText.textContent = "Live stream connected, waiting for the first upload";
        return;
      }

      gallery.innerHTML = images.map((img) => `
        <div class="card">
          <a href="${img.url}">
            <img class="thumb" src="${img.url}?t=${img.modified}" alt="${img.name}">
          </a>
          <div class="meta">
            <div class="name">${img.label}</div>
            <a class="btn" href="${img.url}">Open</a>
          </div>
        </div>
      `).join("");

      gallery.classList.remove("hidden");
      emptyState.classList.add("hidden");
      statusText.textContent = `Live stream connected, ${images.length} poster${images.length === 1 ? "" : "s"} loaded`;
    }

    const posterStream = new EventSource("/events/posters");

    posterStream.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      renderCards(payload.images);
    };

    posterStream.onopen = () => {
      statusText.textContent = "Live stream connected";
    };

    posterStream.onerror = () => {
      statusText.textContent = "Live stream reconnecting";
    };
  </script>
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
    images = list_generated_images()
    return render_template_string(
        INDEX_HTML,
        images=images,
        input_dir=str(INPUT_DIR),
        interval=SCAN_INTERVAL_SECONDS,
    )


@app.get("/api/posters")
def posters_api():
    return jsonify(
        {
            "images": list_generated_images(),
            "input_dir": str(INPUT_DIR),
            "interval": SCAN_INTERVAL_SECONDS,
        }
    )


@app.get("/events/posters")
def posters_events():
    return Response(
        stream_with_context(stream_poster_updates()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/generated/<path:filename>")
def generated_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    template_path = WANTED_TEMPLATE_PATH
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found at: {WANTED_TEMPLATE_PATH}\n"
            "Fix WANTED_TEMPLATE_PATH to point to your wanted poster image on your PC."
        )

    print(f"[INFO] Watching folder: {INPUT_DIR.resolve()}")
    print(f"[INFO] Output folder:   {OUTPUT_DIR.resolve()}")
    print("[INFO] Open: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
