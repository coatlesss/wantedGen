# picToWebsite — Design Doc
## wildWest Image → Website Gallery Integration

---

## How the Current System Works

1. `ping.py` runs as a ZMQ REP server on Jetson B (port 5555)
2. Jetson A sends a JPEG image over ethernet via ZMQ
3. `ping.py` decodes it and saves to `ReceivedImages/img_<timestamp>.jpg`
4. **Nothing else happens yet** — the image sits in ReceivedImages/ unprocessed

The Website (`wantedGen.py`, Flask port 5000) is on the **same machine** as Jetson B.
It serves a live SSE gallery that auto-displays any `.jpg` written to
`Website/static/generated/`.

---

## Goal

After a new image lands in `ReceivedImages/`, automatically:
1. Detect the face in the image
2. Identify which suspect it is (CJ, Cameron, or Tolga)
3. Increment that suspect's bounty by $100
4. Generate an updated wanted poster (parchment style, mustache, bounty text)
5. Save the poster to `Website/static/generated/<name>.jpg`
6. Website SSE stream picks it up → gallery card updates live

---

## Architecture

```
Jetson A  ──ZMQ──►  ping.py  ──saves──►  ReceivedImages/img_<ts>.jpg
                                                    │
                                         picwebtest.py (polling loop)
                                                    │
                                    ┌───────────────┤
                                    │               │
                               Haar cascade      SFace + suspects.pkl
                               face detection    identify suspect
                                    │               │
                                    └───────────────┤
                                                    │
                                           make_poster()
                                    (from poster_service.py)
                                                    │
                                 Website/static/generated/<name>.jpg
                                                    │
                                         Flask SSE stream
                                                    │
                                         Browser gallery ✓
```

---

## New Script: `picwebtest.py`

`ping.py` stays untouched. `picwebtest.py` is a standalone polling loop that
runs alongside it.

### Responsibilities
- Load models once at startup (Haar cascade, SFace, suspects.pkl, mustache asset)
- Poll `ReceivedImages/` every 2 seconds for `.jpg` files not yet processed
- For each new file:
  1. Read full frame from disk
  2. Run Haar cascade on full frame → get face bounding box (x, y, w, h)
  3. **Two separate crops from the same detection:**
     - **Recognition crop** — resize the raw face box to 112×112, run `recognizer.feature()` to get SFace embedding (same as `onePic.py`)
     - **Poster crop** — take the face box with `FACE_PADDING` (40px) added on all sides, clamped to frame bounds (same as `poster_service.py`'s `detect_and_crop_face()`). This is the larger padded crop that fills the poster's face box nicely.
  4. Cosine similarity of recognition embedding vs suspects.pkl → identify top match
  5. If score ≥ 0.35 (THRESH): use that suspect's name; else use "unknown_outlaw"
  6. Increment bounty by $100 for that suspect (in-memory dict)
  7. Call `make_poster(poster_crop_bgr, suspect_id, bounty)` — pass the **padded poster crop**, not the 112×112 one
  8. Save poster as JPEG to `Website/static/generated/<suspect_id>.jpg`
  9. Mark file as processed (track in a set)

### Why two crops?

| Crop | Size | Purpose |
|---|---|---|
| Recognition crop | 112×112 | SFace needs this fixed size to extract the embedding for matching |
| Poster crop | padded, ~face box + 40px | Fills the 480×480 face zone in the poster — a tight crop looks bad at that scale |

### Key reuse from existing files

| Logic | Source file | What to reuse |
|---|---|---|
| Face detection | `onePic.py` | Haar cascade `detectMultiScale()` |
| Recognition crop | `onePic.py` | `cv2.resize(frame[y:y+h, x:x+w], (112, 112))` → `recognizer.feature()` |
| Poster crop | `poster_service.py` | `detect_and_crop_face()` padding logic (`FACE_PADDING = 40`) |
| SFace match | `onePic.py` | Cosine similarity loop vs `suspects.pkl` |
| Poster generation | `poster_service.py` | `make_poster()`, `_load_mustache_rgba()`, all constants |
| Bounty tracking | `poster_service.py` | Same `BOUNTIES` dict pattern |

### Paths (same machine)

```python
BASE_DIR     = Path(__file__).resolve().parent          # wildWest/
INPUT_DIR    = BASE_DIR / "ReceivedImages"
WEBSITE_OUT  = BASE_DIR.parent.parent / "Website" / "static" / "generated"

SFACE_MODEL  = BASE_DIR / "face_recognition_sface_2021dec.onnx"
SUSPECTS_PKL = BASE_DIR / "suspects.pkl"
MUSTACHE_PNG = BASE_DIR / "assets" / "mustache.png"
```

### Poster output behavior

- Same suspect → **overwrites** previous poster file (same filename)
- The Website uses `?t=<mtime>` cache-busting, so the updated poster
  shows immediately in the gallery without a page refresh
- Bounty resets to 0 on script restart (in-memory only — acceptable for demo)

---

## Files Modified

| File | Change |
|---|---|
| `JetsonB/wildWest/picwebtest.py` | **New** — the integration script (currently empty) |
| `ping.py` | None — untouched |
| `poster_service.py` | None — functions reused via import or copy |
| `Website/wantedGen.py` | None — existing SSE pipeline handles new files automatically |

---

## How to Run

```bash
# Terminal 1 — receive images from Jetson A
cd JetsonB/wildWest
python3 ping.py

# Terminal 2 — process received images → update Website gallery
cd JetsonB/wildWest
python3 picwebtest.py

# Terminal 3 — Website (if not already running)
cd Website
python3 wantedGen.py
```

---

## Future Work — Live Video Pipeline

> **Not part of current implementation — planned for a later phase.**

The goal is to replace the single-image ZMQ approach with a continuous live video
stream from Jetson A to Jetson B, running the full pipeline in real time.

### Proposed Flow

```
Jetson A (camera)
  → stream frames over ethernet (ZMQ or GStreamer)
  → Jetson B receives frame buffer
      → extract frames at N fps
      → Haar cascade face detection on each frame
      → if face found:
          → recognition crop (112×112) → SFace → identify suspect
          → poster crop (padded) → make_poster()
          → increment bounty if same person re-detected (with cooldown)
          → overwrite Website/static/generated/<name>.jpg
      → Website SSE stream → gallery updates live
```

### Key Considerations for Later

- **Frame rate**: Don't process every frame — sample every N frames or only when
  a face is detected to avoid hammering the pipeline
- **Debounce / cooldown**: Same suspect shouldn't re-trigger a bounty increment
  on every frame — add a per-suspect cooldown timer (e.g. 15s), same pattern as
  `perception_service.py`'s `--cooldown` arg
- **Transport**: ZMQ PUSH/PULL (already used in ping.py ecosystem) is a natural
  fit; GStreamer is an option if lower latency is needed
- **Stability**: Frame extraction should be robust to dropped frames and
  reconnects — wrap in a retry loop

---

## Verification

1. Start all three processes above
2. Manually copy any test image into `ReceivedImages/`:
   ```bash
   cp test.png ReceivedImages/test_manual.jpg
   ```
3. Within ~2 seconds, confirm:
   - Terminal 2 logs the identified suspect + bounty amount
   - `Website/static/generated/<name>.jpg` is created/updated
   - Website gallery at `http://localhost:5000` shows the new/updated poster card
4. Copy the same image again (or re-fire from Jetson A) — confirm bounty increments by $100
