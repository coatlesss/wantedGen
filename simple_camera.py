#!/usr/bin/env python3

import cv2
import sys

def main():
    cap = None
    
    # Try different camera sources in order
    pipelines = [
        ("Simple /dev/video0", cv2.VideoCapture(0)),
        ("Simple /dev/video1", cv2.VideoCapture(1)),
    ]
    
    # GStreamer pipelines for Jetson
    gst_pipelines = [
        ("nvarguscamerasrc (optimized)", 
         "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"),
        ("v4l2src /dev/video0",
         "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink"),
    ]
    
    # Try simple VideoCapture first
    for name, test_cap in pipelines:
        print(f"Trying: {name}")
        if test_cap.isOpened():
            cap = test_cap
            print(f"✓ Success with {name}\n")
            break
        test_cap.release()
    
    # Try GStreamer pipelines if simple didn't work
    if cap is None:
        for name, pipeline in gst_pipelines:
            print(f"Trying: {name}")
            test_cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if test_cap.isOpened():
                cap = test_cap
                print(f"✓ Success with {name}\n")
                break
            test_cap.release()
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open camera with any method")
        print("\nTroubleshooting:")
        print("1. Check camera connection: ls -la /dev/video*")
        print("2. Check if camera is in use: fuser /dev/video0")
        print("3. Verify OpenCV has GStreamer support:")
        print("   python3 -c \"import cv2; print(cv2.getBuildInformation())\" | grep GStreamer")
        return
    
    print("=" * 50)
    print("Camera opened successfully!")
    print("Press 'q' to quit")
    print("=" * 50)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Display the frame
            cv2.imshow("Jetson Camera", frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed")

if __name__ == "__main__":
    main()
