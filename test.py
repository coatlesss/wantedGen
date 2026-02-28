import cv2
import os

# Check what video devices are available
print("Checking available video devices...")
for i in range(10):
    device = f"/dev/video{i}"
    if os.path.exists(device):
        print(f"  Found: {device}")

print("\nAttempting to open camera...")

# For Jetson CSI camera, use nvarguscamerasrc
JETSON_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
    "appsink"
)

cap = cv2.VideoCapture(JETSON_PIPELINE, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera")
    print("\nTroubleshooting:")
    print("- Make sure a CSI camera is connected to your Jetson")
    print("- Try: gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink")
    print("- Or check: v4l2-ctl --list-devices")
    quit()

print("Camera opened successfully!")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    
    cv2.imshow("Camera Feed", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()