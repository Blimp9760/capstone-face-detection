"""Image in, information about every face out."""

import json
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

DATASET = Path(__file__).resolve().parent / "dataset"
THRESHOLD = 0.45  # Starting value, not a guarantee of identity.
VERSION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODELS = {
    "face_detection_yunet": "face_detection_yunet_2023mar.onnx",
    "face_recognition_sface": "face_recognition_sface_2021dec.onnx",
}


def cosine(a, b):
    """Compare two face feature vectors; similarity is not a probability."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    length = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / length) if length else -1.0


class Brain:
    def __init__(self):
        folder = DATASET / "models"
        folder.mkdir(parents=True, exist_ok=True)
        for model_folder, filename in MODELS.items():
            file = folder / filename
            if not file.exists():
                print(f"Downloading {filename}...")
                url = (f"https://media.githubusercontent.com/media/opencv/opencv_zoo/"
                       f"{VERSION}/models/{model_folder}/{filename}")
                temporary = file.with_suffix(".part")
                try:
                    urlretrieve(url, temporary)
                    temporary.replace(file)
                finally:
                    temporary.unlink(missing_ok=True)

        self.detector = cv2.FaceDetectorYN.create(
            str(folder / MODELS["face_detection_yunet"]), "", (320, 320), score_threshold=0.65)
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(folder / MODELS["face_recognition_sface"]), "")

        known_file = DATASET / "known_people.json"
        self.people = json.loads(known_file.read_text(encoding="utf-8")) if known_file.exists() else {}

    def analyze(self, image):
        """Return boxes, face detection scores, names, and numerical features."""
        if image is None or image.size == 0:
            return []

        height, width = image.shape[:2]
        scale = max(1.0, 360 / min(height, width))
        working = cv2.resize(image, None, fx=scale, fy=scale) if scale > 1 else image
        self.detector.setInputSize((working.shape[1], working.shape[0]))
        _, detected = self.detector.detect(working)
        results = []

        for face in detected if detected is not None else []:
            box = [round(float(v) / scale) for v in face[:4]]
            features = None
            try:
                aligned = self.recognizer.alignCrop(working, face)
                features = self.recognizer.feature(aligned).ravel().astype(np.float32)
            except cv2.error:
                pass

            name, best_score = "Unknown", None
            if features is not None and self.people:
                matches = [(cosine(features, stored), person)
                           for person, photos in self.people.items() for stored in photos]
                if matches:
                    best_score, closest = max(matches)
                    if best_score >= THRESHOLD:
                        name = closest

            results.append({
                "box": box,                  # x, y, width, height
                "confidence": float(face[14]),  # Detection confidence only
                "name": name,
                "similarity": best_score,
                "features": features,        # Used when enrolling/training
            })
        return results
