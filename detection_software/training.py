"""Download LFW photos, train an SVM, and record real test results."""

import csv
from datetime import datetime, timezone
from pathlib import Path
import random

import cv2
import joblib
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC

from detection_software.faces import FaceFinder, file_hash, similarity

THRESHOLDS = [0.363, 0.700, 0.850, 0.900, 0.950]


def download_dataset(data):
    print("Downloading/checking the LFW dataset...")
    fetch_lfw_people(data_home=str(data), min_faces_per_person=0,
                     funneled=True, resize=0.1, slice_=None)
    return data / "lfw_home" / "lfw_funneled"


def choose_photos(folder):
    """Use different photos for training and tests; unknown people stay separate."""
    randomizer = random.Random(42)
    people = [(p.name, list(p.glob("*.jpg"))) for p in folder.iterdir()
              if p.is_dir() and len(list(p.glob("*.jpg"))) >= 16]
    people.sort(key=lambda p: (-len(p[1]), p[0]))
    if len(people) < 10:
        raise RuntimeError("LFW did not contain enough people with 16+ photographs")
    selected = people[:10]
    randomizer.shuffle(selected)

    photos = []
    for index, (name, files) in enumerate(selected):
        files = sorted(files)
        randomizer.shuffle(files)
        if index < 6:  # People the classifier is allowed to learn
            groups = {"train": files[:10], "validation": files[10:13], "test": files[13:16]}
        elif index < 8:  # Completely unknown people for choosing the threshold
            groups = {"validation": files[:8]}
        else:  # Different unknown people, saved for the final test
            groups = {"test": files[:8]}
        for part, images in groups.items():
            for image in images:
                photos.append({"name": name, "known": index < 6, "split": part,
                               "path": str(image), "sha256": file_hash(image)})
    return photos


def train(data):
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    model_file = data / "trained.joblib"
    if model_file.exists():
        print("Model already trained. Delete dataset/trained.joblib to train again.")
        return

    photos = choose_photos(download_dataset(data))
    finder = FaceFinder(data)
    print(f"Reading {len(photos)} face photographs...")
    for number, photo in enumerate(photos, 1):
        image = cv2.imread(photo["path"])
        face = finder.first(image) if image is not None else None
        photo["vector"] = face["vector"] if face else None
        photo["faces_found"] = finder.faces_found if image is not None else 0
        if number % 25 == 0:
            print(f"  {number}/{len(photos)}")

    known = [p for p in photos if p["known"] and p["split"] == "train" and p["vector"] is not None]
    labels = {p["name"] for p in known}
    if len(labels) < 2:
        raise RuntimeError("Too few usable training images")
    model = SVC(kernel="linear", C=1.0, random_state=42)
    model.fit(np.array([p["vector"] for p in known]), [p["name"] for p in known])
    examples = {name: [p["vector"] for p in known if p["name"] == name] for name in labels}

    joblib.dump({"model": model, "examples": examples, "photos": photos}, model_file)
    print(f"Trained on {len(known)} photos of {len(labels)} people. Saved in dataset/trained.joblib")


def guess(saved, vector, threshold):
    """The SVM picks a name, then cosine similarity decides Known or Unknown."""
    if vector is None:
        return "No Face", None
    name = str(saved["model"].predict(vector.reshape(1, -1))[0])
    score = max(similarity(vector, example) for example in saved["examples"][name])
    return (name if score >= threshold else "Unknown"), score


def check(saved, photo, threshold):
    expected = photo["name"] if photo["known"] else "Unknown"
    observed, score = guess(saved, photo["vector"], threshold)
    return {"expected": expected, "observed": observed, "similarity": score,
            "threshold": threshold, "faces_found": photo["faces_found"],
            "file": photo["path"], "sha256": photo["sha256"]}


def mistakes(rows):
    # Wrong person, false acceptance, and missed recognition all count as errors.
    return sum(r["observed"] != r["expected"] for r in rows)


def save_csv(file, rows):
    with open(file, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else ["result"])
        writer.writeheader()
        writer.writerows(rows)


def learning_log_tests(data, saved, threshold, video=None, video_label=None):
    """T01-T08 use real images; a missing video is always marked Not Run."""
    photos = saved["photos"]
    known = next(p for p in photos if p["known"] and p["split"] == "train" and p["vector"] is not None)
    other = [p for p in photos if p["name"] == known["name"] and p["split"] == "test"]
    unknown = []
    for p in photos:
        if p["split"] == "test" and not p["known"] and p["name"] not in [x["name"] for x in unknown]:
            unknown.append(p)

    finder = FaceFinder(data)
    cases = []

    def record(test_id, image, expected, source, note):
        face = finder.first(image) if image is not None else None
        observed, score = guess(saved, face["vector"] if face else None, threshold)
        cases.append({"test": test_id, "expected": expected, "observed": observed,
                      "similarity": score, "threshold": threshold, "faces_found": finder.faces_found,
                      "result": "Pass" if observed == expected else "Fail",
                      "source": source, "note": note, "time_utc": datetime.now(timezone.utc).isoformat()})

    def not_run(test_id, reason):
        cases.append({"test": test_id, "expected": "", "observed": "Not Run",
                      "similarity": None, "threshold": threshold, "faces_found": None,
                      "result": "Not Run", "source": "", "note": reason,
                      "time_utc": datetime.now(timezone.utc).isoformat()})

    # T01: same training photo. This is NOT an independent accuracy test.
    record("T01", cv2.imread(known["path"]), known["name"], known["sha256"], "Self-match")
    if other:
        image = cv2.imread(other[0]["path"])
        record("T02", image, known["name"], other[0]["sha256"], "Independent photo")
        record("T03", (image * 0.6).astype(np.uint8), known["name"],
               other[0]["sha256"], "Digital brightness 60%, not a real dark room")
        if len(other) > 1:
            record("T04", cv2.imread(other[1]["path"]), known["name"],
                   other[1]["sha256"], "Independent photo; pose not verified")
        else:
            not_run("T04", "No second photo")
        covered = image.copy()
        first = finder.first(image)
        if first:
            x, y = first["eyes"].mean(axis=0).astype(int)
            w, h = first["box"][2:]
            cv2.rectangle(covered, (max(0, x - w // 2), max(0, y - h // 10)),
                          (min(covered.shape[1], x + w // 2), min(covered.shape[0], y + h // 10)),
                          (0, 0, 0), -1)
        record("T05", covered, known["name"], other[0]["sha256"], "Artificial eye cover")
    else:
        for test_id in ("T02", "T03", "T04", "T05"):
            not_run(test_id, "No independent test image")

    for test_id, person in zip(("T06", "T07"), unknown):
        record(test_id, cv2.imread(person["path"]), "Unknown", person["sha256"],
               "Different unknown LFW person")
    for test_id in ("T06", "T07"):
        if test_id not in [r["test"] for r in cases]:
            not_run(test_id, "Not enough unknown identities")

    if video and video_label:
        clip = cv2.VideoCapture(str(video))
        image = None
        try:
            while clip.isOpened():
                ok, frame = clip.read()
                if not ok:
                    break
                if finder.first(frame):
                    image = frame
                    break
        finally:
            clip.release()
        if image is not None:
            record("T08", image, video_label, file_hash(video), "Real video frame")
        else:
            not_run("T08", "Video could not be opened or face not found")
    else:
        not_run("T08", "No real video supplied")
    return sorted(cases, key=lambda row: row["test"])


def test(data, video=None, video_label=None):
    data = Path(data)
    file = data / "trained.joblib"
    if not file.exists():
        raise FileNotFoundError("No model yet. Run: py -3.13 run.py")
    saved = joblib.load(file)  # Only open the trained model made by this program.

    validation = [p for p in saved["photos"] if p["split"] == "validation"]
    scores = [(threshold, mistakes([check(saved, p, threshold) for p in validation]))
              for threshold in THRESHOLDS]
    threshold = min(scores, key=lambda item: (item[1], item[0]))[0]
    rows = [check(saved, p, threshold) for p in saved["photos"] if p["split"] == "test"]
    cases = learning_log_tests(data, saved, threshold, video, video_label)
    save_csv(data / "test_results.csv", rows)
    save_csv(data / "learning_log_tests.csv", cases)
    save_csv(data / "thresholds.csv", [{"threshold": t, "validation_errors": errors} for t, errors in scores])
    correct = len(rows) - mistakes(rows)
    print(f"\nFinal test: {correct}/{len(rows)} correct at threshold {threshold}")
    print(f"Results are in the dataset folder (T01-T08, thresholds, and test_results).")
    print("Small LFW sample only. Scores are not identity probabilities.")
    return rows, cases
