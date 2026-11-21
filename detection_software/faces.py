"""Find faces with YuNet and compare their SFace features."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

# Fixed versions from the official OpenCV Zoo project.
VERSION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODELS = {
    "yunet": ("face_detection_yunet", "face_detection_yunet_2023mar.onnx",
              "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "sface": ("face_recognition_sface", "face_recognition_sface_2021dec.onnx",
              "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}


def file_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def download_models(data):
    folder = Path(data) / "models"
    folder.mkdir(parents=True, exist_ok=True)
    for name, (subfolder, filename, expected) in MODELS.items():
        path = folder / filename
        if path.exists() and file_hash(path) == expected:
            continue
        path.unlink(missing_ok=True)
        print(f"Downloading {name}...", flush=True)
        urls = [
            f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{VERSION}/models/{subfolder}/{filename}",
            f"https://huggingface.co/opencv/opencv_zoo/resolve/main/models/{subfolder}/{filename}",
        ]
        for url in urls:
            try:
                temporary = path.with_suffix(".part")
                with urlopen(url, timeout=120) as response, open(temporary, "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if file_hash(temporary) != expected:
                    raise ValueError("Model checksum did not match")
                temporary.replace(path)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
        if not path.exists():
            raise RuntimeError(f"Could not download {name}. Check the internet connection.")
    return folder


def similarity(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if not np.linalg.norm(a) or not np.linalg.norm(b):
        raise ValueError("Empty face features")
    return float(np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1, 1))


class FaceFinder:
    def __init__(self, data):
        folder = download_models(data)
        self.detector = cv2.FaceDetectorYN.create(
            str(folder / MODELS["yunet"][1]), "", (320, 320), score_threshold=0.65
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(folder / MODELS["sface"][1]), "")
        self.faces_found = 0

    def find(self, image):
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        self.faces_found = 0 if faces is None else len(faces)
        results = []
        if faces is not None:
            for face in faces:
                try:
                    cropped = self.recognizer.alignCrop(image, face)
                    vector = self.recognizer.feature(cropped).ravel().astype(np.float32)
                    if not np.linalg.norm(vector):
                        continue
                    vector /= np.linalg.norm(vector)
                    results.append({
                        "box": [int(n) for n in face[:4]],
                        "eyes": np.asarray(face[4:8]).reshape(2, 2),
                        "vector": vector,
                    })
                except cv2.error:
                    continue
        return results

    def first(self, image):
        faces = self.find(image)
        return max(faces, key=lambda x: x["box"][2] * x["box"][3]) if faces else None
