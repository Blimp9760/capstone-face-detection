"""The brain: find a face, turn it into numbers, and compare it with saved faces."""

from pathlib import Path
from urllib.request import urlopen
import hashlib

import cv2
import numpy as np

# Both models are pretrained by OpenCV. My project trains a separate classifier.
OPENCV_VERSION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODELS = {
    "detector": (
        "face_detection_yunet", "face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "recognizer": (
        "face_recognition_sface", "face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def hash_file(path):
    hash_value = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def cosine(a, b):
    """Closer to 1 means more similar; this is NOT a percentage of certainty."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    length = np.linalg.norm(a) * np.linalg.norm(b)
    if length == 0:
        return -1.0
    return float(np.clip(np.dot(a, b) / length, -1, 1))


class Brain:
    def __init__(self, dataset_folder):
        self.folder = Path(dataset_folder) / "models"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.download_models()

        detector_path = self.folder / MODELS["detector"][1]
        recognizer_path = self.folder / MODELS["recognizer"][1]
        self.detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.65)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def download_models(self):
        for name, (source_folder, filename, expected_hash) in MODELS.items():
            path = self.folder / filename
            if path.exists() and hash_file(path) == expected_hash:
                continue

            path.unlink(missing_ok=True)
            url = (f"https://media.githubusercontent.com/media/opencv/opencv_zoo/"
                   f"{OPENCV_VERSION}/models/{source_folder}/{filename}")
            print(f"Downloading {name} model...")
            temporary = path.with_suffix(".part")
            try:
                with urlopen(url, timeout=120) as response, open(temporary, "wb") as output:
                    while piece := response.read(1024 * 1024):
                        output.write(piece)
                if hash_file(temporary) != expected_hash:
                    raise RuntimeError(f"The {name} download failed its file check")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    def see(self, image):
        """Take an OpenCV image and return each face's box, score, and feature vector."""
        if image is None or image.size == 0:
            return []

        height, width = image.shape[:2]
        # LFW photographs are small. A slightly larger copy helps the detector.
        scale = max(1.0, 360.0 / min(height, width))
        working_image = (cv2.resize(image, None, fx=scale, fy=scale)
                         if scale > 1 else image)
        work_h, work_w = working_image.shape[:2]
        self.detector.setInputSize((work_w, work_h))
        _, detections = self.detector.detect(working_image)

        found = []
        for face in detections if detections is not None else []:
            x, y, w, h = (int(round(number / scale)) for number in face[:4])
            confidence = float(face[14])  # How sure YuNet is that this is a face.

            try:
                aligned = self.recognizer.alignCrop(working_image, face)
                features = self.recognizer.feature(aligned).ravel().astype(np.float32)
                if not np.linalg.norm(features):
                    features = None
            except cv2.error:
                features = None

            found.append({
                "box": [x, y, w, h],        # Left, top, width, height (pixels)
                "confidence": confidence,  # Face DETECTION confidence, not identity
                "features": features,      # Numbers describing this face
            })
        return found

    def recognize(self, image, saved_people=None, threshold=0.363):
        """Add a name and similarity to each face, if any faces have been enrolled."""
        saved_people = saved_people or {}
        faces = self.see(image)

        for face in faces:
            face["name"] = "Unknown"
            face["similarity"] = None
            if face["features"] is None:
                continue

            best_score = -1.0
            best_name = "Unknown"
            for name, examples in saved_people.items():
                for example in examples:
                    score = cosine(face["features"], example)
                    if score > best_score:
                        best_score, best_name = score, name

            if best_score >= -1.0 and saved_people:
                face["similarity"] = best_score
                if best_score >= threshold:
                    face["name"] = best_name
        return faces
