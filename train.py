"""Save people from your own photos, or run an optional LFW training experiment."""

import csv
import json
import random
import sys
from pathlib import Path

import cv2
from brain import Brain, DATASET, cosine


def enroll():
    """Each subfolder of dataset/people is the name of one person."""
    root = DATASET / "people"
    root.mkdir(parents=True, exist_ok=True)
    brain = Brain()
    people = {}

    for person in sorted(root.iterdir()):
        if not person.is_dir():
            continue
        features = []
        for photo in sorted(person.iterdir()):
            if photo.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            image = cv2.imread(str(photo))
            faces = brain.analyze(image)
            if len(faces) == 1 and faces[0]["features"] is not None:
                features.append(faces[0]["features"].tolist())
            else:
                print(f"Skipped {photo.name}: need exactly one visible face")
        if features:
            people[person.name] = features
            print(f"{person.name}: {len(features)} photos saved")

    (DATASET / "known_people.json").write_text(json.dumps(people), encoding="utf-8")
    print(f"Saved {len(people)} people. Restart run.py to use them.")


def train_lfw():
    """Optional experiment: train an SVM and test it on different LFW photos."""
    import joblib
    from sklearn.datasets import fetch_lfw_people
    from sklearn.svm import SVC

    print("Downloading LFW dataset if needed...")
    fetch_lfw_people(data_home=str(DATASET), min_faces_per_person=0,
                     funneled=True, resize=0.1, slice_=None)
    root = DATASET / "lfw_home" / "lfw_funneled"
    people = [(p.name, sorted(p.glob("*.jpg"))) for p in root.iterdir() if p.is_dir()]
    people = sorted((p for p in people if len(p[1]) >= 16), key=lambda p: -len(p[1]))[:10]
    if len(people) < 10:
        raise RuntimeError("Not enough LFW identities with 16 photographs")

    randomizer = random.Random(42)
    randomizer.shuffle(people)
    brain = Brain()
    photos = []
    for i, (name, files) in enumerate(people):
        randomizer.shuffle(files)
        if i < 6:  # Known people: separate training, validation and test photos.
            sets = {"train": files[:10], "validation": files[10:13], "test": files[13:16]}
        elif i < 8:  # Unknown identities for choosing the threshold.
            sets = {"validation": files[:8]}
        else:  # Different unknown identities held back for the final test.
            sets = {"test": files[:8]}
        for group, paths in sets.items():
            for path in paths:
                faces = brain.analyze(cv2.imread(str(path)))
                face = max(faces, key=lambda f: f["box"][2] * f["box"][3]) if faces else None
                photos.append((name if i < 6 else "Unknown", group, path,
                               face["features"] if face else None))

    training = [(name, vector) for name, group, _, vector in photos
                if group == "train" and vector is not None]
    if len({name for name, _ in training}) < 2:
        raise RuntimeError("Not enough detectable faces to train the classifier")
    model = SVC(kernel="linear", C=1, random_state=42)
    model.fit([vector for _, vector in training], [name for name, _ in training])
    examples = {name: [v for n, v in training if n == name] for name, _ in training}

    def predict(vector, threshold):
        if vector is None:
            return "No Face", None
        name = str(model.predict([vector])[0])
        score = max(cosine(vector, other) for other in examples[name])
        return (name if score >= threshold else "Unknown"), score

    thresholds = [0.363, 0.5, 0.7, 0.85, 0.9, 0.95]
    validation = [(name, vector) for name, group, _, vector in photos if group == "validation"]
    threshold = min(thresholds, key=lambda t: sum(predict(v, t)[0] != name
                                                  for name, v in validation))

    # Final test photographs were not used to train or choose the threshold.
    results = []
    for expected, group, path, vector in photos:
        if group == "test":
            actual, score = predict(vector, threshold)
            results.append({"photo": str(path), "expected": expected, "actual": actual,
                            "similarity": score, "threshold": threshold,
                            "correct": actual == expected})

    joblib.dump(model, DATASET / "lfw_svm.joblib")
    with (DATASET / "lfw_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    correct = sum(row["correct"] for row in results)
    print(f"LFW experiment: {correct}/{len(results)} correct, threshold {threshold}.")
    print("Saved dataset/lfw_svm.joblib and dataset/lfw_results.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "lfw":
        train_lfw()
    else:
        enroll()
