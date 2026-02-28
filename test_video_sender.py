#!/usr/bin/env python3
"""
Test video sender - simulates Jetson A sending frames to receivefeed.py
Sends frames from a webcam or test pattern to the receiver.
"""
import zmq
import cv2
import numpy as np
import time
import sys

# Config
JETSON_B_IP = "127.0.0.1"  # localhost for testing
LIVE_PORT = 5556
TOPIC = "yolo"

def create_test_frame(counter):
    """Generate a test pattern frame with timestamp."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Color gradient background
    for y in range(480):
        for x in range(640):
            frame[y, x] = [
                int((x / 640) * 255),
                int((y / 480) * 255),
                128
            ]
    
    # Add text
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"TEST FEED - Frame {counter}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, timestamp, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Camera Active", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add a moving box
    box_x = int((counter % 100) * 6.4)
    box_y = int(((counter // 2) % 100) * 4.8)
    cv2.rectangle(frame, (box_x, box_y), (box_x + 50, box_y + 50), (0, 255, 255), -1)
    
    return frame

def main():
    print(f"[SENDER] Connecting to {JETSON_B_IP}:{LIVE_PORT}")
    
    ctx = zmq.Context()
    pub_sock = ctx.socket(zmq.PUB)
    pub_sock.connect(f"tcp://{JETSON_B_IP}:{LIVE_PORT}")
    
    print("[SENDER] Waiting 1 second for connection to establish...")
    time.sleep(1)
    
    # Try to use webcam first, fall back to test pattern
    cap = None
    use_camera = False
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                use_camera = True
                print("[SENDER] Using webcam")
            else:
                cap.release()
                cap = None
    except:
        pass
    
    if not use_camera:
        print("[SENDER] No webcam detected, using test pattern")
    
    counter = 0
    fps_start = time.time()
    fps_counter = 0
    
    try:
        while True:
            if use_camera and cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[SENDER] Camera read failed, switching to test pattern")
                    use_camera = False
                    cap.release()
                    cap = None
                    continue
                
                # Add overlay to camera feed
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"LIVE - Frame {counter}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, timestamp, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Generate test pattern
                frame = create_test_frame(counter)
            
            # Encode as JPEG
            ret, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                print("[SENDER] Failed to encode frame")
                continue
            
            jpg_bytes = jpg.tobytes()
            
            # Send via ZMQ PUB
            pub_sock.send_multipart([TOPIC.encode(), jpg_bytes])
            
            counter += 1
            fps_counter += 1
            
            # Print FPS every 2 seconds
            now = time.time()
            if now - fps_start >= 2.0:
                fps = fps_counter / (now - fps_start)
                print(f"[SENDER] Sending at {fps:.1f} fps, frame {counter}")
                fps_counter = 0
                fps_start = now
            
            # Send at ~10 fps
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n[SENDER] Stopping sender...")
    finally:
        if cap is not None:
            cap.release()
        pub_sock.close()
        ctx.term()

if __name__ == "__main__":
    main()
