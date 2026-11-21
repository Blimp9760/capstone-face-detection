"""Download the two OpenCV models and turn pictures into face features."""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

# YuNet detects faces. SFace turns aligned faces into vectors of numbers.
MODEL_VERSION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODELS = {
    "yunet": ("face_detection_yunet", "face_detection_yunet_2023mar.onnx",
              "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "sface": ("face_recognition_sface", "face_recognition_sface_2021dec.onnx",
              "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_models(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, (github_folder, filename, expected_hash) in MODELS.items():
        path = folder / filename
        if path.is_file() and file_hash(path) == expected_hash:
            continue
        print(f"Downloading {name} from OpenCV...", flush=True)
        path.unlink(missing_ok=True)
        part = folder / (filename + ".part")
        locations = [
            f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_VERSION}/models/{github_folder}/{filename}",
            f"https://raw.githubusercontent.com/opencv/opencv_zoo/{MODEL_VERSION}/models/{github_folder}/{filename}",
            f"https://huggingface.co/opencv/opencv_zoo/resolve/main/models/{github_folder}/{filename}",
        ]
        failures = []
        for url in locations:
            try:
                request = Request(url, headers={"User-Agent": "FaceCapstone/1.0"})
                with urlopen(request, timeout=120) as response, open(part, "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if file_hash(part) != expected_hash:
                    raise ValueError("Downloaded file did not match the official model hash")
                part.replace(path)
                break
            except Exception as error:
                failures.append(str(error))
                part.unlink(missing_ok=True)
        if not path.is_file():
            raise RuntimeError(f"Cannot download {name}. Check your internet connection: {failures}")
    return folder


def normalize(numbers):
    numbers = np.asarray(numbers, dtype=np.float32).reshape(-1)
    size = float(np.linalg.norm(numbers))
    if size == 0:
        raise ValueError("An empty face vector cannot be compared")
    return numbers / size


def similarity(first, second):
    """Cosine similarity: larger scores generally mean more similar faces."""
    return float(np.clip(np.dot(normalize(first), normalize(second)), -1, 1))


class FaceDetector:
    def __init__(self, folder):
        folder = Path(folder)
        self.detector = cv2.FaceDetectorYN.create(
            str(folder / MODELS["yunet"][1]), "", (320, 320), score_threshold=0.65
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(folder / MODELS["sface"][1]), "")
        self.faces_found = 0

    def find_faces(self, image):
        if image is None or image.size == 0:
            raise ValueError("No image was provided")
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, found = self.detector.detect(image)
        self.faces_found = 0 if found is None else len(found)
        result = []
        if found is None:
            return result

        for face in found:
            try:
                aligned = self.recognizer.alignCrop(image, face)
                features = normalize(self.recognizer.feature(aligned))
            except (cv2.error, ValueError):
                continue
            box = tuple(int(round(number)) for number in face[:4])
            eyes = np.asarray(face[4:8], dtype=np.float32).reshape(2, 2)
            result.append({"box": box, "features": features, "eyes": eyes})
        return result

    def find_one(self, image):
        faces = self.find_faces(image)
        return max(faces, key=lambda item: item["box"][2] * item["box"][3]) if faces else None


class SavedFaces:
    """Only save face features on this computer, not the actual photos."""

    def __init__(self, path):
        self.path = Path(path)
        self.people = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.people, indent=2), encoding="utf-8")

    def add(self, name, features):
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError("Please enter a name between 1 and 100 characters")
        self.people.setdefault(name, []).append(normalize(features).tolist())
        self.save()

    def delete(self, name):
        if name in self.people:
            del self.people[name]
            self.save()
            return True
        return False

    def recognize(self, features, threshold):
        best_name, best_score = "Unknown", -1.0
        for name, examples in self.people.items():
            for saved in examples:
                score = similarity(features, saved)
                if score > best_score:
                    best_name, best_score = name, score
        if best_score < threshold:
            return "Unknown", None if best_score == -1.0 else best_score
        return best_name, best_score
