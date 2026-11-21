"""My face-recognition project: train, test, or open the camera.

The actual face-finding code lives in brain.py.
All downloads and results go in the dataset folder.
"""

import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import cv2
    import joblib
    import numpy as np
    from sklearn.datasets import fetch_lfw_people
    from sklearn.svm import SVC
    from brain import Brain, cosine, hash_file
except ImportError as error:
    print(f"Missing Python library: {error}")
    print("Install these yourself: py -3.13 -m pip install numpy opencv-python scikit-learn joblib")
    sys.exit(1)


DATASET = Path(__file__).resolve().parent / "dataset"
MODEL_FILE = DATASET / "trained.joblib"
SAVED_FACES = DATASET / "saved_faces.json"
THRESHOLDS = [0.363, 0.700, 0.850, 0.900, 0.950]


# --------------------- DATASET AND TRAINING ---------------------

def choose_photos(folder):
    """Six known people, two unknown for validation, two unknown for testing."""
    randomizer = random.Random(42)
    people = [(p.name, sorted(p.glob("*.jpg"))) for p in folder.iterdir()
              if p.is_dir() and len(list(p.glob("*.jpg"))) >= 16]
    people.sort(key=lambda person: (-len(person[1]), person[0]))
    if len(people) < 10:
        raise RuntimeError("LFW needs at least 10 people with 16 photographs each")
    selected = people[:10]
    randomizer.shuffle(selected)

    photos = []
    for index, (name, pictures) in enumerate(selected):
        randomizer.shuffle(pictures)
        if index < 6:
            groups = {"train": pictures[:10], "validation": pictures[10:13],
                      "test": pictures[13:16]}
        elif index < 8:
            groups = {"validation": pictures[:8]}
        else:
            groups = {"test": pictures[:8]}

        for split, files in groups.items():
            for file in files:
                photos.append({"person": name, "known": index < 6,
                               "split": split, "file": str(file),
                               "sha256": hash_file(file)})
    return photos


def train():
    DATASET.mkdir(exist_ok=True)
    if MODEL_FILE.exists():
        print("Already trained. Delete dataset/trained.joblib if you want to retrain.")
        return

    print("Downloading the LFW photo dataset (only needed the first time)...")
    # We use the original JPG files, not scikit-learn's tiny resized array.
    fetch_lfw_people(data_home=str(DATASET), resize=0.1, slice_=None)
    photos = choose_photos(DATASET / "lfw_home" / "lfw_funneled")
    brain = Brain(DATASET)

    for number, photo in enumerate(photos, 1):
        image = cv2.imread(photo["file"])
        faces = brain.see(image)
        largest = max(faces, key=lambda f: f["box"][2] * f["box"][3]) if faces else None
        photo["features"] = largest["features"] if largest else None
        photo["found"] = len(faces)
        photo["confidence"] = largest["confidence"] if largest else None
        if number % 25 == 0:
            print(f"Processed {number}/{len(photos)} photos")

    training = [p for p in photos if p["split"] == "train" and p["features"] is not None]
    if len({p["person"] for p in training}) < 2:
        raise RuntimeError("Too few detectable training faces")

    classifier = SVC(kernel="linear", C=1.0)
    classifier.fit([p["features"] for p in training],
                   [p["person"] for p in training])

    examples = {}
    for photo in training:
        examples.setdefault(photo["person"], []).append(photo["features"])

    joblib.dump({"classifier": classifier, "examples": examples, "photos": photos}, MODEL_FILE)
    print(f"Trained an SVM using {len(training)} photographs. Saved: {MODEL_FILE}")


# -------------------------- TESTING ----------------------------

def predict(saved, features, threshold):
    """The SVM proposes a person; cosine similarity accepts/rejects it."""
    if features is None:
        return "No Face", None
    name = str(saved["classifier"].predict(np.asarray(features).reshape(1, -1))[0])
    score = max(cosine(features, example) for example in saved["examples"][name])
    return (name if score >= threshold else "Unknown"), score


def make_result(saved, photo, threshold):
    expected = photo["person"] if photo["known"] else "Unknown"
    name, score = predict(saved, photo["features"], threshold)
    return {"expected": expected, "result": name, "similarity": score,
            "threshold": threshold, "faces_found": photo["found"],
            "confidence": photo["confidence"], "source_file": photo["file"],
            "sha256": photo["sha256"]}


def write_csv(filename, rows):
    if not rows:
        return
    with open(DATASET / filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def learning_log_tests(brain, saved, threshold, video=None, video_label=None):
    """Eight experiments. If there's no real video, T08 is marked Not Run."""
    photos = saved["photos"]
    enrolled = next(p for p in photos if p["split"] == "train" and p["features"] is not None)
    other_photos = [p for p in photos if p["split"] == "test"
                    and p["person"] == enrolled["person"]]
    unknown_photos = []
    for photo in photos:
        if photo["split"] == "test" and not photo["known"]:
            if photo["person"] not in [p["person"] for p in unknown_photos]:
                unknown_photos.append(photo)

    results = []

    def record(number, image, expected, source, notes):
        faces = brain.see(image)
        face = max(faces, key=lambda f: f["box"][2] * f["box"][3]) if faces else None
        name, score = predict(saved, face["features"] if face else None, threshold)
        results.append({
            "test": number, "expected": expected, "result": name,
            "faces_found": len(faces), "confidence": face["confidence"] if face else None,
            "similarity": score, "threshold": threshold,
            "status": "Pass" if name == expected else "Fail",
            "source": source, "notes": notes,
            "time_utc": datetime.now(timezone.utc).isoformat(),
        })

    def skip(number, why):
        results.append({"test": number, "expected": "", "result": "Not Run",
                        "faces_found": "", "confidence": "", "similarity": "",
                        "threshold": threshold, "status": "Not Run", "source": "",
                        "notes": why, "time_utc": datetime.now(timezone.utc).isoformat()})

    record("T01", cv2.imread(enrolled["file"]), enrolled["person"],
           enrolled["sha256"], "Self-match; not independent testing")

    if other_photos:
        test_image = cv2.imread(other_photos[0]["file"])
        record("T02", test_image, enrolled["person"], other_photos[0]["sha256"],
               "Independent photograph")
        record("T03", (test_image * 0.6).astype(np.uint8), enrolled["person"],
               other_photos[0]["sha256"], "60% digital brightness, not real low light")

        if len(other_photos) >= 2:
            record("T04", cv2.imread(other_photos[1]["file"]), enrolled["person"],
                   other_photos[1]["sha256"], "Separate photo; pose change unverified")
        else:
            skip("T04", "No second independent photo")

        covered = test_image.copy()
        found = brain.see(test_image)
        if found:
            x, y, w, h = max(found, key=lambda f: f["box"][2] * f["box"][3])["box"]
            cv2.rectangle(covered, (max(0, x), max(0, y + h // 4)),
                          (min(covered.shape[1], x + w), min(covered.shape[0], y + h // 2)),
                          (0, 0, 0), -1)
        record("T05", covered, enrolled["person"], other_photos[0]["sha256"],
               "Digital eye-region cover; might fail before recognition")
    else:
        for number in ("T02", "T03", "T04", "T05"):
            skip(number, "No independent photo")

    for number, photo in zip(("T06", "T07"), unknown_photos):
        record(number, cv2.imread(photo["file"]), "Unknown", photo["sha256"],
               "Different unknown LFW identity")
    for number in ("T06", "T07"):
        if number not in [r["test"] for r in results]:
            skip(number, "Not enough unknown identities")

    if video and video_label:
        capture = cv2.VideoCapture(str(video))
        image = None
        try:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break
                if brain.see(frame):
                    image = frame
                    break
        finally:
            capture.release()
        if image is not None:
            record("T08", image, video_label, hash_file(video), "Real video frame")
        else:
            skip("T08", "No usable frame in video")
    else:
        skip("T08", "No prerecorded video provided")

    return sorted(results, key=lambda row: row["test"])


def test(video=None, video_label=None):
    if not MODEL_FILE.exists():
        print("No trained model yet. Run: py -3.13 run.py train")
        return
    # Only open your own model file. Never open an untrusted .joblib download.
    saved = joblib.load(MODEL_FILE)
    validation = [p for p in saved["photos"] if p["split"] == "validation"]
    score_table = []

    for threshold in THRESHOLDS:
        rows = [make_result(saved, p, threshold) for p in validation]
        errors = sum(r["expected"] != r["result"] for r in rows)
        score_table.append({"threshold": threshold, "validation_errors": errors,
                            "validation_total": len(rows)})
    chosen = min(score_table, key=lambda row: (row["validation_errors"], row["threshold"]))
    threshold = chosen["threshold"]

    final_rows = [make_result(saved, p, threshold) for p in saved["photos"]
                  if p["split"] == "test"]
    brain = Brain(DATASET)
    cases = learning_log_tests(brain, saved, threshold, video, video_label)

    write_csv("thresholds.csv", score_table)
    write_csv("test_results.csv", final_rows)
    write_csv("learning_log_tests.csv", cases)
    correct = sum(r["expected"] == r["result"] for r in final_rows)
    print(f"Final test: {correct}/{len(final_rows)} correct at similarity threshold {threshold}")
    print("See dataset/ for the actual CSV results (including T01-T08).")
    print("This small dataset is only a classroom experiment, not population-wide accuracy.")


# ---------------------- WEBCAM PROGRAM ------------------------

def load_people():
    if SAVED_FACES.exists():
        return json.loads(SAVED_FACES.read_text(encoding="utf-8"))
    return {}


def save_people(people):
    SAVED_FACES.write_text(json.dumps(people, indent=2), encoding="utf-8")


def camera():
    DATASET.mkdir(exist_ok=True)
    brain = Brain(DATASET)
    people = load_people()
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise RuntimeError("Could not open webcam")

    print("E = save face | D = delete face | Q = quit")
    try:
        while True:
            ok, image = webcam.read()
            if not ok:
                break

            faces = brain.recognize(image, people)
            for face in faces:
                x, y, w, h = face["box"]
                score = face["similarity"]
                score_text = f"similarity {score:.2f}" if score is not None else "no match"
                label = f"{face['name']} | {score_text}"
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(image, label, (x, max(22, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

            cv2.putText(image, "Local processing | E: enroll | D: delete | Q: quit",
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow("My Face Recognition Project", image)
            key = cv2.waitKey(1) & 255
            if key in (ord("q"), 27):
                break

            if key == ord("e") and faces:
                biggest = max(faces, key=lambda f: f["box"][2] * f["box"][3])
                if biggest["features"] is None:
                    print("Could not extract face features. Try another angle.")
                    continue
                name = input("Name to remember: ").strip()
                permission = input("Do they agree to saving their face features? (yes/no): ")
                if name and permission.lower().strip() == "yes":
                    people.setdefault(name, []).append(biggest["features"].tolist())
                    save_people(people)
                    print(f"Saved {name}")

            if key == ord("d"):
                name = input("Name to delete: ").strip()
                if name in people and input("Type DELETE to confirm: ") == "DELETE":
                    del people[name]
                    save_people(people)
                    print(f"Deleted {name}")
    finally:
        webcam.release()
        cv2.destroyAllWindows()


# -------------------------- START HERE ------------------------

def main():
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
    else:
        print("\n1 - Open camera")
        print("2 - Train using LFW photos")
        print("3 - Run the tests")
        choice = input("Choose 1, 2, or 3: ").strip()

    if choice in ("1", "camera"):
        camera()
    elif choice in ("2", "train"):
        train()
    elif choice in ("3", "test"):
        video = sys.argv[2] if len(sys.argv) > 2 else None
        label = sys.argv[3] if len(sys.argv) > 3 else None
        test(video, label)
    else:
        print("Use 'camera', 'train', or 'test'.")


if __name__ == "__main__":
    main()
