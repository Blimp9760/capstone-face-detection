"""Train my own identity classifier with LFW photos, then test it."""

import csv
import json
import sys
import random
from pathlib import Path

import cv2
import joblib
import numpy as np

from brain import Brain, cosine

DATASET = Path(__file__).resolve().parent / "dataset"
THRESHOLDS = [0.363, 0.5, 0.7, 0.85, 0.9, 0.95]


def choose_photos(folder):
    """Same people for training/testing; different people for unknown testing."""
    people = []
    for folder_for_person in folder.iterdir():
        if folder_for_person.is_dir():
            photos = sorted(folder_for_person.glob("*.jpg"))
            if len(photos) >= 16:
                people.append((folder_for_person.name, photos))

    people.sort(key=lambda person: (-len(person[1]), person[0]))
    if len(people) < 10:
        raise RuntimeError("Not enough LFW identities with 16+ photos")

    randomizer = random.Random(42)
    chosen = people[:10]
    randomizer.shuffle(chosen)
    selected = []
    for i, (name, photos) in enumerate(chosen):
        randomizer.shuffle(photos)
        if i < 6:  # Six people the SVM is allowed to recognize
            splits = {"train": photos[:10], "validation": photos[10:13], "test": photos[13:16]}
        elif i < 8:  # Different unknown people, just for setting the threshold
            splits = {"validation": photos[:8]}
        else:  # Two other unknown people, only used for the final test
            splits = {"test": photos[:8]}

        for split, files in splits.items():
            for file in files:
                selected.append({"name": name, "known": i < 6, "split": split, "file": file})
    return selected


def guess(classifier, examples, features, threshold):
    if features is None:
        return "No Face", None
    name = str(classifier.predict([features])[0])
    score = max(cosine(features, ref) for ref in examples[name])
    return (name if score >= threshold else "Unknown"), score


def test_row(photo, classifier, examples, threshold):
    expected = photo["name"] if photo["known"] else "Unknown"
    actual, score = guess(classifier, examples, photo["features"], threshold)
    return {
        "file": str(photo["file"]), "expected": expected, "actual": actual,
        "similarity": score, "threshold": threshold,
        "passed": actual == expected,
    }


def save_csv(filename, rows):
    if not rows:
        return
    with open(DATASET / filename, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def learning_log_tests(finder, photos, classifier, examples, threshold):
    """Eight Learning Log cases. T08 is Not Run without a real video."""
    first = next(p for p in photos if p["known"] and p["split"] == "train" and p["features"] is not None)
    same = [p for p in photos if p["known"] and p["name"] == first["name"] and p["split"] == "test"]
    unknown = []
    for photo in photos:
        if photo["split"] == "test" and not photo["known"] and photo["name"] not in [p["name"] for p in unknown]:
            unknown.append(photo)

    rows = []

    def check(number, image, expected, note):
        found = finder.analyze(image)
        face = max(found, key=lambda f: f["box"][2] * f["box"][3]) if found else None
        actual, score = guess(classifier, examples, face["features"] if face else None, threshold)
        rows.append({"test": number, "expected": expected, "actual": actual,
                     "faces_found": len(found), "similarity": score, "threshold": threshold,
                     "result": "Pass" if actual == expected else "Fail", "note": note})

    check("T01", cv2.imread(str(first["file"])), first["name"], "Training image compared with itself")
    if same:
        image = cv2.imread(str(same[0]["file"]))
        check("T02", image, first["name"], "Independent photograph")
        check("T03", (image * 0.6).astype(np.uint8), first["name"], "Digital brightness 60%, not real low light")
        second = same[1] if len(same) > 1 else same[0]
        check("T04", cv2.imread(str(second["file"])), first["name"], "Another photograph; pose not verified")
        covered = image.copy()
        faces = finder.analyze(image)
        if faces:
            x, y, w, h = max(faces, key=lambda f: f["box"][2] * f["box"][3])["box"]
            cv2.rectangle(covered, (max(0, x), max(0, y + h // 4)),
                          (min(covered.shape[1], x + w), min(covered.shape[0], y + h // 2)), (0, 0, 0), -1)
        check("T05", covered, first["name"], "Artificial eye-area cover")
    for number, photo in zip(("T06", "T07"), unknown):
        check(number, cv2.imread(str(photo["file"])), "Unknown", "Different unknown LFW person")
    rows.append({"test": "T08", "expected": "", "actual": "Not Run", "faces_found": "",
                 "similarity": "", "threshold": threshold, "result": "Not Run",
                 "note": "No real video supplied"})
    save_csv("learning_log_tests.csv", sorted(rows, key=lambda row: row["test"]))




def add_my_people():
    """Read photos in dataset/people/Name/ and remember their face features."""
    folder = DATASET / "people"
    folder.mkdir(parents=True, exist_ok=True)
    brain = Brain(DATASET, use_training=False)
    saved = {}

    for person in sorted(folder.iterdir()):
        if not person.is_dir():
            continue

        photos = [p for p in sorted(person.iterdir())
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        vectors = []
        for photo in photos:
            image = cv2.imread(str(photo))
            faces = brain.analyze(image) if image is not None else []
            if len(faces) == 1 and faces[0]["features"] is not None:
                vectors.append(faces[0]["features"].tolist())
            else:
                print(f"Skipped {photo.name}: use a photo with exactly one face")

        if vectors:
            saved[person.name] = vectors
            print(f"Added {person.name}: {len(vectors)} photos")
            if len(vectors) < 3:
                print("  Tip: 3-5 different photos usually work better.")

    output = DATASET / "known_people.json"
    output.write_text(json.dumps(saved), encoding="utf-8")
    print(f"Saved {len(saved)} people to {output}")
    print("Now run run.py to see their names. Other people should say Unknown.")
    if not saved:
        print("No people saved. Put photos into dataset/people/PersonName/ first.")

def main():
    from sklearn.datasets import fetch_lfw_people
    from sklearn.svm import SVC
    DATASET.mkdir(exist_ok=True)
    print("Downloading LFW if it is not already saved...")
    fetch_lfw_people(data_home=str(DATASET), min_faces_per_person=0,
                     funneled=True, resize=0.1, slice_=None)
    photos = choose_photos(DATASET / "lfw_home" / "lfw_funneled")
    brain = Brain(DATASET, use_training=False)

    print(f"Processing {len(photos)} images...")
    for i, photo in enumerate(photos, 1):
        image = cv2.imread(str(photo["file"]))
        faces = brain.analyze(image)
        biggest = max(faces, key=lambda f: f["box"][2] * f["box"][3]) if faces else None
        photo["features"] = biggest["features"] if biggest else None
        if i % 20 == 0:
            print(f"{i}/{len(photos)}")

    training = [p for p in photos if p["split"] == "train" and p["features"] is not None]
    classifier = SVC(kernel="linear", C=1, random_state=42)
    classifier.fit([p["features"] for p in training], [p["name"] for p in training])
    examples = {name: [p["features"] for p in training if p["name"] == name]
                for name in sorted({p["name"] for p in training})}

    validation = [p for p in photos if p["split"] == "validation"]
    scores = []
    for threshold in THRESHOLDS:
        rows = [test_row(p, classifier, examples, threshold) for p in validation]
        scores.append({"threshold": threshold, "errors": sum(not row["passed"] for row in rows)})
    save_csv("thresholds.csv", scores)
    threshold = min(scores, key=lambda item: (item["errors"], item["threshold"]))["threshold"]

    joblib.dump({"classifier": classifier, "examples": examples, "threshold": threshold},
                DATASET / "trained.joblib")
    print(f"Training finished. Chosen similarity threshold: {threshold}")

    test_photos = [p for p in photos if p["split"] == "test"]
    results = [test_row(p, classifier, examples, threshold) for p in test_photos]
    save_csv("results.csv", results)
    correct = sum(row["passed"] for row in results)
    print(f"Final test: {correct}/{len(results)} correct. This is a small LFW experiment.")
    learning_log_tests(brain, photos, classifier, examples, threshold)
    print("Reports saved in dataset/. Run run.py to use your trained model.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "people":
        add_my_people()
    else:
        main()
