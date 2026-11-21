"""The brain of my project: image in, face information out."""

from pathlib import Path
from urllib.request import urlopen
import hashlib
import json

import cv2
import numpy as np

# These two AI models were already trained by OpenCV.
VERSION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODELS = {
    "yunet": ("face_detection_yunet", "face_detection_yunet_2023mar.onnx",
              "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "sface": ("face_recognition_sface", "face_recognition_sface_2021dec.onnx",
              "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}


def file_hash(path):
    sha = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def cosine(a, b):
    """Cosine similarity of two face vectors, not a probability."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    length = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / length) if length else -1.0


class Brain:
    def __init__(self, dataset="dataset", use_training=True):
        self.dataset = Path(dataset)
        model_folder = self.dataset / "models"
        model_folder.mkdir(parents=True, exist_ok=True)

        for kind, (folder, filename, expected_hash) in MODELS.items():
            path = model_folder / filename
            if path.exists() and file_hash(path) == expected_hash:
                continue

            path.unlink(missing_ok=True)
            url = (f"https://media.githubusercontent.com/media/opencv/opencv_zoo/"
                   f"{VERSION}/models/{folder}/{filename}")
            temporary = path.with_suffix(".part")
            print(f"Downloading {kind} model...")
            try:
                with urlopen(url, timeout=120) as response, open(temporary, "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if file_hash(temporary) != expected_hash:
                    raise RuntimeError(f"The downloaded {kind} model failed its checksum")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

        self.detector = cv2.FaceDetectorYN.create(
            str(model_folder / MODELS["yunet"][1]), "", (320, 320), score_threshold=0.65)
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(model_folder / MODELS["sface"][1]), "")

        # Photos of people you added yourself are stored as face vectors.
        # If you have enrolled people, they take priority over the LFW demo.
        self.people = {}
        people_file = self.dataset / "known_people.json"
        if use_training and people_file.exists():
            self.people = json.loads(people_file.read_text(encoding="utf-8"))

        self.training = None
        trained_file = self.dataset / "trained.joblib"
        if use_training and not self.people and trained_file.exists():
            import joblib
            # Only load a training file that this project created locally.
            saved = joblib.load(trained_file)
            if {"classifier", "examples", "threshold"}.issubset(saved):
                self.training = saved
            else:
                print("Old training file found. Run train.py to replace it.")

    def analyze(self, image):
        """Return one dictionary per detected face. The input is an OpenCV image."""
        if image is None or image.size == 0:
            return []

        height, width = image.shape[:2]
        scale = max(1.0, 360 / min(height, width))
        working = cv2.resize(image, None, fx=scale, fy=scale) if scale > 1 else image
        self.detector.setInputSize((working.shape[1], working.shape[0]))
        _, detections = self.detector.detect(working)
        results = []

        for face in detections if detections is not None else []:
            x, y, w, h = [round(float(number) / scale) for number in face[:4]]
            result = {
                "box": [x, y, w, h],          # x, y, width, height (pixels)
                "confidence": float(face[14]), # Does this look like a face?
                "name": "Unknown",
                "similarity": None,           # How similar to a known person?
                "features": None,             # Numerical face description
            }

            try:
                aligned = self.recognizer.alignCrop(working, face)
                features = self.recognizer.feature(aligned).flatten().astype(np.float32)
                if np.linalg.norm(features):
                    result["features"] = features
            except cv2.error:
                pass

            if result["features"] is not None:
                features = result["features"]
                if self.people:
                    # Compare with everyone you have added from your own photos.
                    scores = [(cosine(features, saved), name)
                              for name, photos in self.people.items()
                              for saved in photos]
                    if scores:
                        score, name = max(scores)
                        result["similarity"] = score
                        if score >= 0.45:  # Starting point; adjust after testing.
                            result["name"] = name
                elif self.training is not None:
                    # Otherwise use the SVM trained on the LFW dataset.
                    name = str(self.training["classifier"].predict([features])[0])
                    score = max(cosine(features, ref)
                                for ref in self.training["examples"][name])
                    result["similarity"] = score
                    if score >= self.training["threshold"]:
                        result["name"] = name

            results.append(result)
        return results
