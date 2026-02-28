#!/usr/bin/env python3
"""
picwebtest.py — wildWest → Website integration

Polls ReceivedImages/ for new JPEG files from ping.py, runs the full pipeline:
  face detection → recognition crop (112×112) → SFace identify
  → poster crop (padded) → generate wanted poster → save to Website gallery

Run alongside ping.py:
  Terminal 1: python3 ping.py
  Terminal 2: python3 picwebtest.py
"""

import logging
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("picwebtest")

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parent
INPUT_DIR    = BASE_DIR / "ReceivedImages"
WEBSITE_OUT  = BASE_DIR.parent.parent / "Website" / "static" / "generated"

SFACE_MODEL  = str(BASE_DIR / "face_recognition_sface_2021dec.onnx")
SUSPECTS_PKL = BASE_DIR / "suspects.pkl"
MUSTACHE_PNG = BASE_DIR / "assets" / "mustache.png"

FONT_BOLD  = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

# ── Config ────────────────────────────────────────────────────────────────────

POLL_INTERVAL = 2.0   # seconds between ReceivedImages/ scans
MATCH_THRESH  = 0.35  # cosine similarity threshold for suspect match
FACE_PADDING  = 40    # px added around face bbox for poster crop
UNKNOWN_LABEL = "unknown_outlaw"

# ── Poster geometry (mirrors poster_service.py) ───────────────────────────────

POSTER_W   = 800
POSTER_H   = 1000
FACE_BOX_W = 480
FACE_BOX_H = 480
FACE_TOP_Y = 280
FACE_X     = (POSTER_W - FACE_BOX_W) // 2  # 160

# ── Colour palette (mirrors poster_service.py) ────────────────────────────────

PARCHMENT_BG          = (235, 215, 175, 255)
BORDER_COLOR          = (80,   45,  15, 255)
TEXT_DARK             = (40,   20,   5, 255)
TEXT_RED              = (160,  20,  10, 255)
ORNAMENT_COLOR        = (100,  60,  20, 255)
MUSTACHE_BG_THRESHOLD = 180

# ── Load models + assets once at startup ─────────────────────────────────────

log.info("Loading Haar cascade...")
try:
    _HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
except AttributeError:
    _HAAR_XML = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
_cascade = cv2.CascadeClassifier(_HAAR_XML)

log.info("Loading SFace recognizer...")
_recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

log.info("Loading suspects.pkl...")
with open(SUSPECTS_PKL, "rb") as _f:
    SUSPECTS: dict = pickle.load(_f)
log.info("Loaded %d suspect(s): %s", len(SUSPECTS), list(SUSPECTS.keys()))

log.info("Loading mustache asset...")
def _load_mustache_rgba() -> Image.Image:
    img = Image.open(MUSTACHE_PNG).convert("RGBA")
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (r > MUSTACHE_BG_THRESHOLD) & (g > MUSTACHE_BG_THRESHOLD) & (b > MUSTACHE_BG_THRESHOLD)
    arr[mask, 3] = 0
    return Image.fromarray(arr, "RGBA")

MUSTACHE_RGBA = _load_mustache_rgba()

# ── Bounty state (in-memory) ──────────────────────────────────────────────────

BOUNTIES: dict[str, int] = {}

# ── Face detection ────────────────────────────────────────────────────────────

def detect_face(frame_bgr: np.ndarray):
    """
    Run Haar cascade on the full frame.
    Returns (x, y, w, h) of the largest detected face, or None if none found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def make_recognition_crop(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """112×112 resize of the raw face box — what SFace.feature() expects (from onePic.py)."""
    return cv2.resize(frame_bgr[y:y + h, x:x + w], (112, 112))


def make_poster_crop(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Padded face crop for the poster — larger than recognition crop so it fills the 480×480 zone."""
    fh, fw = frame_bgr.shape[:2]
    x1 = max(0, x - FACE_PADDING)
    y1 = max(0, y - FACE_PADDING)
    x2 = min(fw, x + w + FACE_PADDING)
    y2 = min(fh, y + h + FACE_PADDING)
    return frame_bgr[y1:y2, x1:x2]

# ── Suspect identification ────────────────────────────────────────────────────

def identify(rec_crop: np.ndarray) -> tuple[str, float]:
    """
    Extract SFace embedding from a 112×112 BGR crop, compare against suspects.pkl.
    Returns (suspect_id, best_score).
    """
    emb = _recognizer.feature(rec_crop).flatten()
    emb /= np.linalg.norm(emb)

    best_name  = UNKNOWN_LABEL
    best_score = -1.0
    for name, ref_emb in SUSPECTS.items():
        score = float(np.dot(emb, ref_emb))
        log.debug("  similarity[%s] = %.4f", name, score)
        if score > best_score:
            best_score = score
            best_name  = name

    if best_score < MATCH_THRESH:
        log.info("No match above threshold (best=%.3f) → %s", best_score, UNKNOWN_LABEL)
        return UNKNOWN_LABEL, best_score

    log.info("Identified: '%s' (score=%.3f)", best_name, best_score)
    return best_name, best_score

# ── Poster generation (mirrors poster_service.py) ────────────────────────────

def _cv2_to_pil_rgba(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, "RGB").convert("RGBA")


def _draw_centered(draw: ImageDraw.ImageDraw, text: str,
                   font: ImageFont.FreeTypeFont, y: int,
                   fill: tuple, shadow: int = 0) -> None:
    bbox   = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    x      = (POSTER_W - text_w) // 2
    if shadow:
        draw.text((x + shadow, y + shadow), text, font=font, fill=(0, 0, 0, 80))
    draw.text((x, y), text, font=font, fill=fill)


def _ornament_line(draw: ImageDraw.ImageDraw, y: int) -> None:
    m = 45
    draw.line([(m, y),     (POSTER_W - m, y)],     fill=ORNAMENT_COLOR, width=1)
    draw.line([(m, y + 3), (POSTER_W - m, y + 3)], fill=ORNAMENT_COLOR, width=3)
    draw.line([(m, y + 7), (POSTER_W - m, y + 7)], fill=ORNAMENT_COLOR, width=1)


def _apply_mustache(poster: Image.Image) -> None:
    target_w = int(FACE_BOX_W * 0.55)
    scale    = target_w / MUSTACHE_RGBA.width
    target_h = int(MUSTACHE_RGBA.height * scale)
    scaled   = MUSTACHE_RGBA.resize((target_w, target_h), Image.LANCZOS)
    mx = FACE_X + (FACE_BOX_W - target_w) // 2
    my = FACE_TOP_Y + int(FACE_BOX_H * 0.62)
    poster.paste(scaled, (mx, my), scaled)


def make_poster(face_bgr: np.ndarray, suspect_id: str, bounty: int) -> Image.Image:
    """Generate a parchment wanted poster with mustache overlay and bounty text."""
    poster = Image.new("RGBA", (POSTER_W, POSTER_H), PARCHMENT_BG)

    noise = np.random.randint(0, 30, (POSTER_H, POSTER_W, 4), dtype=np.uint8)
    noise[:, :, 3] = 38
    poster = Image.alpha_composite(poster, Image.fromarray(noise, "RGBA"))
    draw   = ImageDraw.Draw(poster)

    # Double-line border with corner dots
    draw.rectangle([10, 10, POSTER_W - 10, POSTER_H - 10], outline=BORDER_COLOR, width=6)
    draw.rectangle([22, 22, POSTER_W - 22, POSTER_H - 22], outline=BORDER_COLOR, width=2)
    for cx, cy in [(10, 10), (POSTER_W - 10, 10), (10, POSTER_H - 10), (POSTER_W - 10, POSTER_H - 10)]:
        draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=BORDER_COLOR)

    # Fonts
    f_wanted   = ImageFont.truetype(FONT_BOLD,  110)
    f_subtitle = ImageFont.truetype(FONT_BOLD,   38)
    f_bounty   = ImageFont.truetype(FONT_BOLD,   52)
    f_small    = ImageFont.truetype(FONT_SERIF,  15)

    _draw_centered(draw, "WANTED",        f_wanted,   y=45,  fill=TEXT_DARK, shadow=3)
    _draw_centered(draw, "DEAD OR ALIVE", f_subtitle, y=158, fill=TEXT_RED)
    _ornament_line(draw, y=205)
    _ornament_line(draw, y=790)
    _ornament_line(draw, y=900)

    # Face photo
    face_pil = _cv2_to_pil_rgba(face_bgr).resize((FACE_BOX_W, FACE_BOX_H), Image.LANCZOS)
    draw.rectangle(
        [FACE_X - 4, FACE_TOP_Y - 4, FACE_X + FACE_BOX_W + 4, FACE_TOP_Y + FACE_BOX_H + 4],
        outline=BORDER_COLOR, width=3,
    )
    poster.paste(face_pil, (FACE_X, FACE_TOP_Y), face_pil)

    _apply_mustache(poster)

    draw = ImageDraw.Draw(poster)
    _draw_centered(draw, f"Bounty: ${bounty:,}", f_bounty, y=825, fill=TEXT_RED, shadow=2)
    _draw_centered(draw, "Wanted by the Marshal's Office", f_small, y=915, fill=ORNAMENT_COLOR)
    _draw_centered(draw, f"[{suspect_id}]", f_small, y=940, fill=ORNAMENT_COLOR)

    return poster

# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_image(img_path: Path) -> None:
    """Full pipeline for one received image: detect → identify → poster → save."""
    log.info("Processing: %s", img_path.name)

    frame = cv2.imread(str(img_path))
    if frame is None:
        log.warning("Could not read image: %s — skipping", img_path.name)
        return

    face = detect_face(frame)
    if face is None:
        log.warning("No face detected in %s — skipping", img_path.name)
        return

    x, y, w, h = face
    log.info("Face detected at (%d, %d) size %dx%d", x, y, w, h)

    rec_crop  = make_recognition_crop(frame, x, y, w, h)   # 112×112 for SFace
    post_crop = make_poster_crop(frame, x, y, w, h)        # padded for poster

    suspect_id, score = identify(rec_crop)

    BOUNTIES[suspect_id] = BOUNTIES.get(suspect_id, 0) + 100
    bounty = BOUNTIES[suspect_id]
    log.info("Suspect: %-16s | Score: %.3f | Bounty: $%d", suspect_id, score, bounty)

    poster_img = make_poster(post_crop, suspect_id, bounty)

    out_path = WEBSITE_OUT / f"{suspect_id}.jpg"
    poster_img.convert("RGB").save(str(out_path), quality=92)
    log.info("Poster saved → %s", out_path)


def main() -> None:
    WEBSITE_OUT.mkdir(parents=True, exist_ok=True)

    # Clear any posters from a previous session
    for old in WEBSITE_OUT.glob("*.jpg"):
        old.unlink()
    log.info("Bounty board cleared — fresh session started")

    # Mark all pre-existing images as already seen — only process NEW arrivals
    processed: set[str] = set()
    if INPUT_DIR.exists():
        processed = {p.name for p in INPUT_DIR.glob("*.jpg")}
        log.info("Skipping %d pre-existing image(s) — waiting for new ones", len(processed))

    log.info("Watching:  %s", INPUT_DIR)
    log.info("Output:    %s", WEBSITE_OUT)
    log.info("Polling every %.1fs — Ctrl+C to stop", POLL_INTERVAL)

    while True:
        if INPUT_DIR.exists():
            for img_path in sorted(INPUT_DIR.glob("*.jpg")):
                if img_path.name in processed:
                    continue
                processed.add(img_path.name)
                try:
                    process_image(img_path)
                except Exception as exc:
                    log.error("Failed on %s: %s", img_path.name, exc)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
