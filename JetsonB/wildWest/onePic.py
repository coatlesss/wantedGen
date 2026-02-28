
import pickle, cv2, numpy as np

SFACE  = "face_recognition_sface_2021dec.onnx"
THRESH = 0.35

# ── change this to any image path ──────────────────────────────────────────
IMG = "/home/clejah/wantedGen/JetsonB/wildWest/CJ/IMG_2927.jpeg"
# ───────────────────────────────────────────────────────────────────────────

recognizer = cv2.FaceRecognizerSF.create(SFACE, "")
cascade    = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

with open("suspects.pkl", "rb") as f:
    suspects = pickle.load(f)

frame = cv2.imread(IMG)
if frame is None:
    print("Could not read image:", IMG); exit(1)

gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    print("No face detected in image")
    exit(0)

x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
face_crop  = cv2.resize(frame[y:y+h, x:x+w], (112, 112))
query_emb  = recognizer.feature(face_crop).flatten()
query_emb /= np.linalg.norm(query_emb)

print(f"\nImage: {IMG}")
print(f"{'Name':<12} {'Score':>8}  {'Match?'}")
print("-" * 35)
for name, ref_emb in suspects.items():
    score = float(np.dot(query_emb, ref_emb))
    match = "<-- MATCH" if score >= THRESH else ""
    print(f"{name:<12} {score:>8.4f}  {match}")
