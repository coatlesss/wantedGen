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

# Pillow <9.1 doesn't have Image.Resampling — fall back to the legacy constant
_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

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
ELEVENLABS_AGENT_ID = "agent_3401kjhc8whcfg9vf8aa0dt716yh"  # Paste your public ElevenLabs agent ID here to enable the Agent tab


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
    face_240 = ImageOps.fit(face_img.convert("RGBA"), (240, 240), method=_LANCZOS)

    # Optional vintage tone (comment out if you want original colors)
    face_240 = ImageOps.colorize(
        ImageOps.grayscale(face_240), black="#2b1a0f", white="#f6e3b2"
    ).convert("RGBA")

    # 2) IMPORTANT: scale that 240x240 up to fill the big PHOTO_BOX (so it isn't tiny)
    l, t, r, b = PHOTO_BOX
    box_w, box_h = (r - l), (b - t)

    # Keep it "photo-like": fit to the box (crop to fill) so it fills the frame.
    face_for_poster = ImageOps.fit(face_240, (box_w, box_h), method=_LANCZOS)

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
      color-scheme: dark;
      --night:#24140b;
      --sand1:#3a2415;
      --sand2:#52331c;
      --paper:#1a130d;
      --paper2:#24180f;
      --ink:#f3e2c6;
      --ink2:#f7d59b;
      --line:#5b3b1f;
      --line-soft:rgba(247,213,155,.14);
      --surface:rgba(32,22,14,.82);
      --surface-strong:rgba(43,28,17,.92);
      --surface-soft:rgba(255,240,214,.05);
      --shadow: 0 18px 55px rgba(0,0,0,.62);
      --radius: 18px;
    }
    *{ box-sizing:border-box; }
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      color: var(--paper);
      background:
        radial-gradient(1200px 600px at 15% -10%, rgba(255,180,90,.30), transparent 65%),
        radial-gradient(900px 500px at 85% 0%, rgba(90,180,255,.18), transparent 60%),
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

    .desert-floor{
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      height: 170px;
      background:
        radial-gradient(700px 120px at 20% 10%, rgba(255,240,205,.18), transparent 60%),
        radial-gradient(900px 140px at 80% 15%, rgba(120,72,28,.12), transparent 58%),
        linear-gradient(180deg, rgba(120,76,28,.08) 0%, rgba(167,113,52,.34) 18%, rgba(198,144,74,.78) 58%, rgba(168,111,48,.96) 100%);
      border-top: 1px solid rgba(86,49,16,.18);
      box-shadow: inset 0 16px 30px rgba(255,228,176,.08);
      pointer-events: none;
      z-index: 0;
    }

    .desert-floor:before{
      content:"";
      position:absolute;
      left:0;
      right:0;
      top:-22px;
      height: 42px;
      background:
        radial-gradient(60px 18px at 4% 100%, rgba(193,138,70,.92), transparent 70%),
        radial-gradient(90px 20px at 14% 100%, rgba(205,149,77,.9), transparent 72%),
        radial-gradient(72px 18px at 27% 100%, rgba(186,131,64,.88), transparent 70%),
        radial-gradient(96px 22px at 41% 100%, rgba(211,158,86,.88), transparent 72%),
        radial-gradient(80px 18px at 56% 100%, rgba(190,136,68,.9), transparent 70%),
        radial-gradient(88px 20px at 71% 100%, rgba(206,152,80,.86), transparent 72%),
        radial-gradient(68px 16px at 86% 100%, rgba(187,132,66,.88), transparent 70%),
        radial-gradient(82px 18px at 97% 100%, rgba(204,149,76,.84), transparent 72%);
      opacity: .95;
    }

    /* cactus png decorations */
    .cactus{
      position: fixed;
      bottom: 6px;
      width: 320px;
      opacity: .52;
      pointer-events:none;
      z-index:0;
      filter: drop-shadow(0 12px 28px rgba(0,0,0,.38)) saturate(1.08) contrast(1.08);
    }
    .cactus.left{ left: 18px; }
    .cactus.right{ right: 18px; transform: scaleX(-1); }

    .tumbleweed{
      position: fixed;
      left: -140px;
      bottom: 26px;
      width: 96px;
      height: 96px;
      pointer-events: none;
      z-index: 0;
      opacity: .88;
      object-fit: contain;
      filter: drop-shadow(0 8px 12px rgba(69,38,12,.26));
      animation: tumble-roll 28s linear infinite;
    }

    @keyframes tumble-roll{
      0%{
        transform: translateX(0) rotate(0deg);
      }
      50%{
        transform: translateX(calc(50vw - 40px)) rotate(540deg);
      }
      100%{
        transform: translateX(calc(100vw + 180px)) rotate(1080deg);
      }
    }

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
      background: linear-gradient(180deg, rgba(48,31,18,.88), rgba(24,16,10,.9));
      border: 1px solid var(--line-soft);
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
      color: #fff4df;
    }
    .sub{
      margin:0;
      color: rgba(243,226,198,.72);
      font-size: 13px;
      line-height: 1.35;
    }

    .badge{
      display:inline-flex;
      align-items:center;
      gap:10px;
      padding:10px 12px;
      border-radius: 999px;
      border: 1px solid rgba(243,226,198,.16);
      background: rgba(8,5,3,.42);
      color: rgba(243,226,198,.88);
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
        radial-gradient(900px 600px at 20% 10%, rgba(255,214,143,.06), transparent 55%),
        radial-gradient(700px 500px at 85% 25%, rgba(0,0,0,.16), transparent 60%),
        linear-gradient(180deg, var(--paper) 0%, var(--paper2) 100%);
      border: 1px solid var(--line-soft);
      box-shadow: 0 25px 70px rgba(0,0,0,.5);
      padding: 18px;
      color: var(--ink);
      position: relative;
      overflow:hidden;
    }
    .paper:before{
      content:"";
      position:absolute; inset:0;
      background:
        radial-gradient(circle at 20% 30%, rgba(255,224,172,.03) 0 2px, transparent 3px),
        radial-gradient(circle at 70% 40%, rgba(255,224,172,.025) 0 1.6px, transparent 3px),
        radial-gradient(circle at 40% 80%, rgba(255,224,172,.025) 0 1.8px, transparent 3px);
      opacity:.7;
      pointer-events:none;
      mix-blend-mode:screen;
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

    .paper-tabs{
      position: relative;
      z-index: 1;
      display:flex;
      gap:10px;
      margin-bottom: 14px;
    }

    .tab-btn{
      border: 1px solid rgba(247,213,155,.10);
      background: var(--surface-soft);
      color: rgba(243,226,198,.72);
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 12px;
      font-weight: 900;
      letter-spacing: .06em;
      text-transform: uppercase;
      cursor: pointer;
      transition: background .15s ease, color .15s ease, transform .15s ease;
    }

    .tab-btn:hover{
      transform: translateY(-1px);
      background: rgba(255,240,214,.10);
    }

    .tab-btn.active{
      background: linear-gradient(180deg, #6b4a28, #4e3218);
      color: #fff4df;
      border-color: rgba(247,213,155,.24);
      box-shadow: 0 8px 18px rgba(0,0,0,.28);
    }

    .tab-panel.hidden{
      display:none;
    }

    .queue{
      font-size: 12px;
      font-weight: 800;
      color: rgba(243,226,198,.66);
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
      border: 1px solid rgba(247,213,155,.10);
      background: var(--surface);
      transition: transform .15s ease, box-shadow .15s ease;
      box-shadow: 0 8px 22px rgba(0,0,0,.24);
    }
    .card:hover{
      transform: translateY(-2px);
      box-shadow: 0 16px 34px rgba(0,0,0,.32);
    }

    .thumb{
      width:100%;
      aspect-ratio: 3/4;
      object-fit: cover;
      display:block;
      background: rgba(255,255,255,.04);
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
      color: #f5dbad;
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
      border: 1px solid rgba(247,213,155,.14);
      background: rgba(255,240,214,.08);
      transition: background .15s ease;
      white-space:nowrap;
    }
    .btn:hover{ background: rgba(255,240,214,.14); }

    .empty{
      padding: 16px;
      border-radius: 14px;
      border: 1px dashed rgba(247,213,155,.22);
      background: rgba(255,240,214,.04);
      color: rgba(243,226,198,.8);
      position:relative;
      z-index:1;
    }

    .agent-shell{
      position: relative;
      z-index: 1;
      min-height: 520px;
      padding-top: 10px;
    }

    .agent-frame{
      display:flex;
      justify-content:center;
      align-items:flex-start;
      min-height: 460px;
    }

    elevenlabs-convai{
      width: 100%;
      max-width: 520px;
      min-height: 460px;
      display: block;
    }

    .story-panel{
      position: relative;
      z-index: 1;
      padding-top: 10px;
      display:grid;
      gap: 14px;
    }

    .story-card{
      border-radius: 16px;
      border: 1px solid rgba(247,213,155,.1);
      background: var(--surface-strong);
      padding: 18px;
      box-shadow: 0 10px 24px rgba(0,0,0,.18);
    }

    .story-kicker{
      margin: 0 0 8px;
      font-size: 11px;
      font-weight: 900;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: rgba(247,213,155,.68);
    }

    .story-card p{
      margin: 0;
      color: rgba(243,226,198,.88);
      line-height: 1.7;
      font-size: 15px;
    }

    .hidden{ display:none; }

    code{
      background: rgba(255,240,214,.06);
      border: 1px solid rgba(247,213,155,.10);
      padding: 2px 6px;
      border-radius: 10px;
      color: var(--ink);
      font-weight: 800;
    }
  </style>
</head>
<body>
  <div class="haze"></div>
  <div class="desert-floor"></div>

  <!-- cactus pngs (put them in static/assets/) -->
  <img class="cactus left"  src="/static/assets/cactus.webp"  alt="cactus left">
  <img class="cactus right" src="/static/assets/cactus.webp" alt="cactus right">
  <img class="tumbleweed" src="/static/assets/tumbleweed.png" alt="" aria-hidden="true">

  <div class="wrap">
    <div class="topbar">
      <div class="brand">
        <h1 class="title">Cowboy Criminal Database</h1>
      </div>
      <div class="badge"><span class="dot"></span> <span id="status-text">Connecting to live upload stream</span></div>
    </div>

    <div class="paper">
      <div class="paper-tabs">
        <button class="tab-btn active" type="button" data-tab="posters">Most Wanted</button>
        <button class="tab-btn" type="button" data-tab="agent">AI Deputy</button>
        <button class="tab-btn" type="button" data-tab="story">Gang Briefing</button>
        <button class="tab-btn" type="button" data-tab="tos">TOS</button>
      </div>

      <section id="tab-posters" class="tab-panel">
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
      </section>

      <section id="tab-agent" class="tab-panel hidden">
        <div class="paper-head">
          <h2>AI Deputy</h2>
          <div class="queue">Powered by ElevenLabs</div>
        </div>

        <div class="agent-shell">
          {% if elevenlabs_agent_id %}
            <div class="agent-frame">
              <elevenlabs-convai agent-id="{{ elevenlabs_agent_id }}" variant="expanded"></elevenlabs-convai>
            </div>
          {% else %}
            <div class="empty">
              Add your ElevenLabs public agent ID to <code>ELEVENLABS_AGENT_ID</code> in this file to enable the embedded agent tab.
            </div>
          {% endif %}
        </div>
      </section>

      <section id="tab-story" class="tab-panel hidden">
        <div class="paper-head">
          <h2>Gang Briefing</h2>
          <div class="queue">For licensed bounty hunters</div>
        </div>

        <div class="story-panel">
          <div class="story-card">
            <div class="story-kicker">Field Notice</div>
            <p>
              Word has spread from the back roads to the student blocks around Clemson: a fast-moving crew known as the
              Three Cactus Thieves has been lifting laptops from dorm rooms, library tables, and unlocked trucks before the dust
              even settles. They travel light, scout in pairs, and leave behind just enough confusion to make witnesses
              doubt what they saw.
            </p>
          </div>

          <div class="story-card">
            <div class="story-kicker">Known Pattern</div>
            <p>
              Their method is simple. One rider watches the foot traffic, another slips in, snatches the machine, and a
              third runs the hardware through a chain of quick handoffs before sunrise. By the time a victim files a
              report, the stolen device is already stripped, wiped, or moving out of county. They favor crowded events,
              late-night study spots, and parking lots near apartment complexes on the edge of town.
            </p>
          </div>

          <div class="story-card">
            <div class="story-kicker">Hunter Guidance</div>
            <p>
              Bounty hunters tracking this gang should keep up to date with the most wanted posters that contain the faces of the notorious gang.
	      The more laptops a criminal has stolen, the higher their bounty. If you need any help, talk to the Deputy and ask about the gang and 
	      where they can be found. Best of luck to all, and be sure to stay safe.
            </p>
          </div>
        </div>
      </section>

      <section id="tab-tos" class="tab-panel hidden">
        <div class="paper-head">
          <h2>Terms of Service</h2>
          <div class="queue">Privacy notice</div>
        </div>

        <div class="story-panel">
          <div class="story-card">
            <div class="story-kicker">Data Handling</div>
            <p>
              Any face images, generated posters, and related data used by this website are stored privately for this
              application only. That data will not be used, shared, or repurposed for any other reason outside the
              operation of this system.
            </p>
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const gallery = document.getElementById("gallery");
    const emptyState = document.getElementById("empty-state");
    const statusText = document.getElementById("status-text");
    const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));
    const tabPanels = {
      posters: document.getElementById("tab-posters"),
      agent: document.getElementById("tab-agent"),
      story: document.getElementById("tab-story"),
      tos: document.getElementById("tab-tos"),
    };

    function setActiveTab(tabName) {
      tabButtons.forEach((button) => {
        button.classList.toggle("active", button.dataset.tab === tabName);
      });

      Object.entries(tabPanels).forEach(([name, panel]) => {
        panel.classList.toggle("hidden", name !== tabName);
      });
    }

    tabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        setActiveTab(button.dataset.tab);
      });
    });

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
  {% if elevenlabs_agent_id %}
    <script src="https://unpkg.com/@elevenlabs/convai-widget-embed" async type="text/javascript"></script>
  {% endif %}
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
        elevenlabs_agent_id=ELEVENLABS_AGENT_ID.strip(),
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

    # Clear the board on every fresh start
    for _old in OUTPUT_DIR.glob("*.jpg"):
        _old.unlink()
    print("[INFO] Bounty board cleared — fresh session")

    print(f"[INFO] Watching folder: {INPUT_DIR.resolve()}")
    print(f"[INFO] Output folder:   {OUTPUT_DIR.resolve()}")
    print("[INFO] Open: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
