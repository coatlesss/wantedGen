#!/usr/bin/env python3
"""
register_suspects.py
One-time setup: reads photos from CJ/, Cameron/, Tolga/ folders,
computes a mean SFace embedding per person, saves to suspects.pkl.

Usage:
  python3 register_suspects.py
"""

import glob
import os
import pickle

import cv2
import numpy as np

SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
OUTPUT_FILE = "suspects.pkl"

# Folder name → suspect_id used as the key sent in /event
SUSPECT_DIRS = {
    "CJ":      "CJ",
    "Cameron": "Cameron",
    "Tolga":   "Tolga",
}

# Haar cascade is built into OpenCV — no separate model file needed
try:
    _HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
except AttributeError:
    _HAAR_XML = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
_CASCADE = cv2.CascadeClassifier(_HAAR_XML)


def compute_embedding(
    recognizer: cv2.FaceRecognizerSF,
    img_bgr: np.ndarray,
) -> np.ndarray | None:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None

    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = img_bgr[y : y + h, x : x + w]
    # SFace expects a 112×112 BGR image
    face_crop = cv2.resize(face_crop, (112, 112))
    feat = recognizer.feature(face_crop)
    return feat.flatten()


def process_folder(
    folder: str,
    recognizer: cv2.FaceRecognizerSF,
) -> np.ndarray | None:
    extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, ext)))

    if not paths:
        print(f"  [!] No images found in {folder}/")
        return None

    embeddings = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [!] Could not read {path}, skipping")
            continue
        emb = compute_embedding(recognizer, img)
        if emb is None:
            print(f"  [!] No face detected in {os.path.basename(path)}, skipping")
        else:
            embeddings.append(emb)

    if not embeddings:
        print(f"  [!] No usable faces found in {folder}/")
        return None

    mean_emb = np.mean(embeddings, axis=0)
    # L2-normalize the mean embedding once
    mean_emb /= np.linalg.norm(mean_emb)
    print(f"  OK — {len(embeddings)}/{len(paths)} photos used")
    return mean_emb


def main() -> None:
    recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

    suspects: dict[str, np.ndarray] = {}

    for folder, suspect_id in SUSPECT_DIRS.items():
        print(f"\nProcessing {folder}/ → '{suspect_id}'")
        if not os.path.isdir(folder):
            print(f"  [!] Directory not found: {folder}/")
            continue
        emb = process_folder(folder, recognizer)
        if emb is not None:
            suspects[suspect_id] = emb

    if not suspects:
        print("\n[ERROR] No suspect embeddings registered. Check folder paths.")
        return

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(suspects, f)

    print(f"\nSaved {len(suspects)} suspect(s) to {OUTPUT_FILE}:")
    for name in suspects:
        print(f"  {name}")


if __name__ == "__main__":
    main()
