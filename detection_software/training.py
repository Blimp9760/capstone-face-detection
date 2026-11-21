"""Download LFW photos and train a small SVM using pretrained SFace features."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import cv2
import joblib
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC

from detection_software.faces import MODELS, FaceDetector, download_models, file_hash


def download_dataset(folder):
    """LFW downloads into dataset/, never into the GitHub repository."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    print("Checking LFW dataset (first download may take a while)...", flush=True)
    fetch_lfw_people(data_home=str(folder), min_faces_per_person=0,
                     resize=0.1, color=False, funneled=True, slice_=None)
    photos = folder / "lfw_home" / "lfw_funneled"
    if not photos.is_dir():
        raise FileNotFoundError(f"LFW images were not found in {photos}")
    return photos


def split_images(paths, randomizer):
    """Keep different pictures of the same known person for each stage."""
    paths = sorted(paths)
    randomizer.shuffle(paths)
    n = len(paths)
    test_size = max(2, n // 5)
    validation_size = max(2, n // 5)
    return {"train": paths[:n - test_size - validation_size],
            "validation": paths[n - test_size - validation_size:n - test_size],
            "test": paths[n - test_size:]}


def prepare_photos(folder, seed=42):
    """Choose 6 known and 6 unknown identities (3 for each evaluation split)."""
    randomizer = random.Random(seed)
    available = []
    for person in folder.iterdir():
        if person.is_dir():
            photos = sorted(person.glob("*.jpg"))
            if len(photos) >= 15:
                available.append((person.name, photos))

    available.sort(key=lambda entry: (-len(entry[1]), entry[0]))
    if len(available) < 12:
        raise RuntimeError("Not enough LFW people with 15+ independent pictures")
    selected = available[:12]
    randomizer.shuffle(selected)
    images = []
    for index, (name, photos) in enumerate(selected):
        label = ("K" if index < 6 else "UV" if index < 9 else "UT") + f"{index + 1:02d}"
        selected_photos = photos.copy()
        randomizer.shuffle(selected_photos)
        selected_photos = selected_photos[:40]
        if index < 6:
            parts = split_images(selected_photos, randomizer)
        else:
            parts = {"validation" if index < 9 else "test": selected_photos}
        for split, files in parts.items():
            for path in files:
                images.append({"label": label, "split": split,
                               "file": str(path.relative_to(folder)), "sha256": file_hash(path)})
    return {"seed": seed, "photos_folder": str(folder), "source": "Labeled Faces in the Wild (LFW)",
            "images": images, "created_utc": datetime.now(timezone.utc).isoformat()}


def train(workspace, force=False):
    workspace = Path(workspace)
    saved_model = workspace / "trained_model.joblib"
    if saved_model.exists() and not force:
        print("Already trained. Using the saved model (run --train to train again).")
        return
    dataset_folder = workspace / "dataset"
    models_folder = download_models(workspace / "models")
    image_folder = download_dataset(dataset_folder)
    manifest = prepare_photos(image_folder)
    dataset_folder.mkdir(exist_ok=True)
    (dataset_folder / "split_info.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    finder = FaceDetector(models_folder)
    samples, vectors, problems = [], [], []
    for i, item in enumerate(manifest["images"], start=1):
        path = image_folder / item["file"]
        if file_hash(path) != item["sha256"]:
            raise RuntimeError(f"A dataset photo changed: {path}")
        image = cv2.imread(str(path))
        if image is None:
            problems.append({**item, "problem": "unreadable image"})
            continue
        face = finder.find_one(image)
        if face is None:
            problems.append({**item, "problem": "face not detected"})
            continue
        samples.append({**item, "face_count": finder.faces_found})
        vectors.append(face["features"])
        if i % 50 == 0:
            print(f"Checking faces: {i}/{len(manifest['images'])}", flush=True)

    known = [i for i, sample in enumerate(samples) if sample["split"] == "train"]
    labels = [samples[i]["label"] for i in known]
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        raise RuntimeError(f"Not enough detected training photos: {dict(counts)}")

    print("Training the SVM classifier...", flush=True)
    model = SVC(kernel="linear", C=1.0, random_state=manifest["seed"])
    model.fit(np.array([vectors[i] for i in known]), labels)
    # The SVM suggests an identity, and cosine similarity decides Known/Unknown.
    examples = {label: np.array([vectors[i] for i in known if samples[i]["label"] == label])
                for label in counts}

    # LFW face embeddings are biometric data; this file must stay local.
    joblib.dump({"classifier": model, "examples": examples,
                 "samples": samples, "vectors": np.array(vectors),
                 "problems": problems, "manifest": manifest,
                 "model_hashes": {name: file_hash(models_folder / details[1])
                                  for name, details in MODELS.items()},
                 "trained_at": datetime.now(timezone.utc).isoformat()}, saved_model)
    print(f"Training complete: {len(counts)} people, {len(known)} photos.")
    print(f"Saved locally at: {workspace / 'trained_model.joblib'}")
