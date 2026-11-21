"""YuNet detect -> SFace alignCrop -> feature embedding -> similarity score."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"

@dataclass
class Face:
    box: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray
    landmarks: np.ndarray


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        raise ValueError("Zero-length face feature")
    return v / norm


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(unit(a), unit(b)), -1, 1))


class FacePipeline:
    def __init__(self, model_dir: Path = Path("models"), detector_score_threshold: float = 0.7):
        for file in (YUNET, SFACE):
            if not (model_dir / file).is_file():
                raise FileNotFoundError(f"Missing {model_dir / file}; run python -m facecap.models")
        self.detector = cv2.FaceDetectorYN.create(str(model_dir / YUNET), "", (320, 320),
                                                   score_threshold=detector_score_threshold, nms_threshold=0.3, top_k=5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(model_dir / SFACE), "")
        self.last_detector_count = 0
        self.last_feature_failures = 0

    def extract_all(self, image: np.ndarray) -> list[Face]:
        if image is None or image.size == 0:
            raise ValueError("Image is missing or empty")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected BGR image with three channels")
        self.last_detector_count = 0
        self.last_feature_failures = 0
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, detections = self.detector.detect(image)
        if detections is None:
            return []
        self.last_detector_count = len(detections)
        results = []
        for det in detections:
            try:
                aligned = self.recognizer.alignCrop(image, det)
                vector = unit(self.recognizer.feature(aligned))
            except (cv2.error, ValueError):
                self.last_feature_failures += 1
                continue
            x, y, bw, bh = (int(round(z)) for z in det[:4])
            results.append(Face((x, y, bw, bh), float(det[14]), vector,
                                np.asarray(det[4:14], dtype=np.float32).reshape(5, 2)))
        return results

    def extract_one(self, image: np.ndarray) -> tuple[np.ndarray | None, int, Face | None]:
        faces = self.extract_all(image)
        if not faces:
            return None, self.last_detector_count, None
        selected = max(faces, key=lambda f: f.box[2] * f.box[3])
        return selected.embedding, self.last_detector_count, selected


def image_from_lfw(rgb: np.ndarray) -> np.ndarray:
    """scikit-learn gives RGB float pixels; models expect BGR uint8."""
    arr = np.asarray(rgb)
    if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.01:
        arr = arr * 255
    return cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
