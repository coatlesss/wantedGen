#!/usr/bin/env python3
"""
Jetson B receiver:
- Receives theft-event suspect crops over ZMQ REP (reliable request/reply).
- Receives live annotated YOLO video over ZMQ SUB (best-effort pub/sub).
"""
import os
import time
from datetime import datetime
import cv2
import numpy as np
import zmq
# ---------------------------
# CONFIG
# ---------------------------
SAVE_DIR = os.environ.get("SAVE_DIR", "evidence_b")
os.makedirs(SAVE_DIR, exist_ok=True)
EVENT_PORT = int(os.environ.get("JETSON_B_PORT", "5555"))
LIVE_PORT = int(os.environ.get("JETSON_B_LIVE_PORT", "5556"))
EVENT_BIND_ADDR = f"tcp://*:{EVENT_PORT}"
LIVE_BIND_ADDR = f"tcp://*:{LIVE_PORT}"
LIVE_TOPIC = os.environ.get("LIVE_STREAM_TOPIC", "yolo")
HEADLESS = (os.environ.get("HEADLESS", "0") == "1") or (os.environ.get("DISPLAY", "") == "")
SAVE_LIVE_VIDEO = os.environ.get("SAVE_LIVE_VIDEO", "0") == "1"
LIVE_VIDEO_FPS = float(os.environ.get("LIVE_VIDEO_FPS", "10"))
LIVE_VIDEO_PATH = os.environ.get("LIVE_VIDEO_PATH", os.path.join(SAVE_DIR, "live_annotated_b.mp4"))

# Live feed snapshot for web viewing
LIVE_SNAPSHOT_ENABLED = os.environ.get("LIVE_SNAPSHOT", "1") == "1"
LIVE_SNAPSHOT_PATH = os.environ.get("LIVE_SNAPSHOT_PATH", "/home/clejah/wantedGen/Website/static/live_feed.jpg")

def try_enable_gui():
    if HEADLESS:
        return False
    try:
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imshow("__test__", dummy)
        cv2.waitKey(1)
        cv2.destroyWindow("__test__")
        return True
    except Exception:
        return False
def decode_jpeg(jpg_bytes):
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
def main():
    gui_available = try_enable_gui()
    if gui_available:
        print("[B] GUI enabled. Showing live stream window.")
    else:
        print("[B] GUI disabled (headless/no OpenCV GUI).")
    ctx = zmq.Context.instance()
    rep_sock = ctx.socket(zmq.REP)
    rep_sock.setsockopt(zmq.LINGER, 2000)
    rep_sock.bind(EVENT_BIND_ADDR)
    sub_sock = ctx.socket(zmq.SUB)
    sub_sock.setsockopt(zmq.LINGER, 0)
    sub_sock.setsockopt(zmq.RCVHWM, 1)
    sub_sock.setsockopt_string(zmq.SUBSCRIBE, LIVE_TOPIC)
    sub_sock.bind(LIVE_BIND_ADDR)
    print(f"[B] Event REP listening on {EVENT_BIND_ADDR}")
    print(f"[B] Live SUB listening on {LIVE_BIND_ADDR} topic='{LIVE_TOPIC}'")
    if LIVE_SNAPSHOT_ENABLED:
        print(f"[B] Live snapshot enabled -> {LIVE_SNAPSHOT_PATH}")
    
    poller = zmq.Poller()
    poller.register(rep_sock, zmq.POLLIN)
    poller.register(sub_sock, zmq.POLLIN)
    writer = None
    live_count = 0
    last_fps_t = time.time()
    try:
        while True:
            events = dict(poller.poll(100))
            if rep_sock in events and events[rep_sock] == zmq.POLLIN:
                data = rep_sock.recv()
                img = decode_jpeg(data)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                if img is not None:
                    out = os.path.join(SAVE_DIR, f"suspect_from_a_{ts}.jpg")
                    cv2.imwrite(out, img)
                    rep_sock.send_string("OK")
                    print(f"[B] Saved suspect crop -> {out}")
                else:
                    rep_sock.send_string("BAD_JPEG")
                    print("[B] Failed to decode suspect JPEG")
            if sub_sock in events and events[sub_sock] == zmq.POLLIN:
                try:
                    topic, jpg = sub_sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    topic, jpg = None, None
                if jpg is not None:
                    frame = decode_jpeg(jpg)
                    if frame is not None:
                        live_count += 1
                        if SAVE_LIVE_VIDEO:
                            if writer is None:
                                h, w = frame.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                writer = cv2.VideoWriter(LIVE_VIDEO_PATH, fourcc, LIVE_VIDEO_FPS, (w, h))
                                print(f"[B] Recording live video -> {LIVE_VIDEO_PATH}")
                            writer.write(frame)
                        
                        # Save latest frame as snapshot for web viewing
                        if LIVE_SNAPSHOT_ENABLED:
                            cv2.imwrite(LIVE_SNAPSHOT_PATH, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        if gui_available:
                            cv2.imshow("Jetson B - Live YOLO Feed", frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key in (27, ord("q")):
                                print("[B] Exit requested from keyboard.")
                                break
            now = time.time()
            if now - last_fps_t >= 2.0:
                fps = live_count / max(1e-6, (now - last_fps_t))
                print(f"[B] Live stream receive rate: {fps:.1f} fps")
                live_count = 0
                last_fps_t = now
    except KeyboardInterrupt:
        print("\n[B] Stopping receiver...")
    finally:
        if writer is not None:
            writer.release()
        if gui_available:
            cv2.destroyAllWindows()
        rep_sock.close(0)
        sub_sock.close(0)
if __name__ == "__main__":
    main()