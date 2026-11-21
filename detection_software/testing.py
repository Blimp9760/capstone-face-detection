"""Check recognition accuracy and save the actual numbers for the Learning Log."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")  # Make graphs without opening a window.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from detection_software.faces import FaceDetector, similarity, file_hash

THRESHOLDS = [0.363, 0.700, 0.850, 0.900, 0.950]


def time_now():
    return datetime.now(timezone.utc).isoformat()


def guess_person(model, vector):
    """SVM chooses a person; we then check that person's best photo match."""
    guess = str(model["classifier"].predict(vector.reshape(1, -1))[0])
    score = max(similarity(vector, saved) for saved in model["examples"][guess])
    return guess, score


def predictions(model, split):
    results = []
    for i, sample in enumerate(model["samples"]):
        if sample["split"] != split:
            continue
        guess, score = guess_person(model, model["vectors"][i])
        results.append({"expected": sample["label"] if sample["label"].startswith("K") else "Unknown",
                        "guess": guess, "similarity": score, "file_hash": sample["sha256"],
                        "faces_found": sample["face_count"], "problem": "", "timestamp": time_now()})
    for sample in model["problems"]:
        if sample["split"] == split:
            results.append({"expected": sample["label"] if sample["label"].startswith("K") else "Unknown",
                            "guess": "No Face", "similarity": None, "file_hash": sample["sha256"],
                            "faces_found": 0, "problem": sample["problem"], "timestamp": time_now()})
    return results


def decide(guess, score, threshold):
    if score is None or guess == "No Face":
        return "No Face"
    return guess if score >= threshold else "Unknown"


def error_counts(rows, threshold):
    false_accept = false_reject = wrong_person = detection_failed = correct = 0
    for row in rows:
        observed = decide(row["guess"], row["similarity"], threshold)
        expected = row["expected"]
        correct += observed == expected
        if observed == "No Face":
            detection_failed += 1
        elif expected == "Unknown" and observed != "Unknown":
            false_accept += 1
        elif expected != "Unknown" and observed == "Unknown":
            false_reject += 1
        elif expected != "Unknown" and observed != expected:
            wrong_person += 1
    return {"threshold": threshold, "photos": len(rows), "correct": correct,
            "false_accepts": false_accept, "false_rejects": false_reject,
            "wrong_person": wrong_person, "detection_failures": detection_failed}


def save_csv(file, records):
    if not records:
        return
    with open(file, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(
            key for record in records for key in record)))
        writer.writeheader()
        writer.writerows(records)


def test_photo(detector, model, image, case_id, expected, source, threshold, note):
    """Check the full detection + feature + recognition pipeline, not just the SVM."""
    face = detector.find_one(image)
    if face is None:
        label, score = "No Face", None
        reason = "YuNet found no face" if detector.faces_found == 0 else "SFace could not extract features"
    else:
        guess, score = guess_person(model, face["features"])
        label = decide(guess, score, threshold)
        reason = "" if label == expected else "The detection or identity decision did not match the expected result"
    return {"case": case_id, "expected": expected, "faces_found": detector.faces_found,
            "similarity": score, "threshold": threshold, "observed": label,
            "result": "Pass" if label == expected else "Fail", "reason": reason,
            "source_hash": source, "processed_image_hash": hashlib.sha256(image.tobytes()).hexdigest(),
            "photo_details": note, "timestamp": time_now()}


def eight_learning_log_tests(workspace, model, threshold, video=None, video_label=None):
    detector = FaceDetector(workspace / "models")
    manifest = model["manifest"]
    root = Path(manifest["photos_folder"])
    tests = []

    def photo(item):
        path = root / item["file"]
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"Source photograph was changed: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot open {path}")
        return image

    def missing(case, expected, reason):
        tests.append({"case": case, "expected": expected, "faces_found": None,
                      "similarity": None, "threshold": threshold, "observed": "Not Run",
                      "result": "Not Run", "reason": reason, "source_hash": "",
                      "processed_image_hash": "", "photo_details": "", "timestamp": time_now()})

    enrolled = next((item for item in manifest["images"]
                     if item["split"] == "train" and item["label"] in model["examples"]), None)
    if enrolled is None:
        raise RuntimeError("No training photograph is available for T01")
    known = enrolled["label"]
    self_picture = photo(enrolled)
    tests.append(test_photo(detector, model, self_picture, "T01", known,
                            enrolled["sha256"], threshold, "Same photo used for enrollment (self-match)"))

    separate_photos = [item for item in manifest["images"]
                       if item["split"] == "test" and item["label"] == known]
    if separate_photos:
        other = photo(separate_photos[0])
        tests.append(test_photo(detector, model, other, "T02", known,
                                separate_photos[0]["sha256"], threshold, "Independent photograph"))
        darker = np.clip(other.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
        tests.append(test_photo(detector, model, darker, "T03", known,
                                separate_photos[0]["sha256"], threshold,
                                "Digitally darkened T02 (60% brightness), not a real dark room"))
        if len(separate_photos) > 1:
            tests.append(test_photo(detector, model, photo(separate_photos[1]), "T04", known,
                                    separate_photos[1]["sha256"], threshold,
                                    "Another independent photo; a different head pose is NOT confirmed"))
        else:
            missing("T04", known, "No second independent testing photograph")
        covered = other.copy()
        found = detector.find_one(other)
        if found:
            x, y = found["eyes"].mean(axis=0).astype(int)
            width = max(14, int(found["box"][2] * 0.85))
            height = max(10, int(found["box"][3] * 0.22))
            left, top = max(0, x - width // 2), max(0, y - height // 2)
        else:
            h, w = other.shape[:2]
            left, top, width, height = w // 4, h // 3, w // 2, h // 6
        cv2.rectangle(covered, (left, top), (min(covered.shape[1], left + width),
                                          min(covered.shape[0], top + height)), (0, 0, 0), -1)
        tests.append(test_photo(detector, model, covered, "T05", known,
                                separate_photos[0]["sha256"], threshold, "Artificial eye-region cover"))
    else:
        for case in ("T02", "T03", "T04", "T05"):
            missing(case, known, "No independent testing photograph")

    unknown_people = sorted({item["label"] for item in manifest["images"]
                             if item["split"] == "test" and item["label"].startswith("UT")})
    for case, unknown in zip(("T06", "T07"), unknown_people):
        item = next(i for i in manifest["images"] if i["label"] == unknown and i["split"] == "test")
        tests.append(test_photo(detector, model, photo(item), case, "Unknown",
                                item["sha256"], threshold, f"Different unknown LFW identity {unknown}"))
    if len(unknown_people) < 2:
        for case in ("T06", "T07"):
            if not any(t["case"] == case for t in tests):
                missing(case, "Unknown", "Fewer than two distinct unknown identities")

    if video and video_label:
        camera = cv2.VideoCapture(str(video))
        frame = None
        try:
            if camera.isOpened():
                for _ in range(60):
                    ok, candidate = camera.read()
                    if not ok:
                        break
                    if detector.find_one(candidate):
                        frame = candidate
                        break
        finally:
            camera.release()
        if frame is not None:
            tests.append(test_photo(detector, model, frame, "T08", video_label,
                                    file_hash(video), threshold, "Actual decoded frame from a video"))
        else:
            missing("T08", video_label, "Video unavailable or no face detected")
    else:
        missing("T08", video_label or "unspecified", "No video and expected identity were provided")
    return sorted(tests, key=lambda test: test["case"])


def make_graphs(folder, results, comparisons):
    figure, axis = plt.subplots(figsize=(7, 4))
    for key, label in (("false_accepts", "False accepts"),
                       ("false_rejects", "False rejects"), ("wrong_person", "Wrong identity")):
        axis.plot([r["threshold"] for r in comparisons], [r[key] for r in comparisons], marker="o", label=label)
    axis.set(xlabel="Similarity threshold", ylabel="Validation images", title="Validation errors")
    axis.legend()
    figure.tight_layout()
    figure.savefig(folder / "threshold_graph.png")
    plt.close(figure)

    labels = sorted({row["expected"] for row in results} | {row["observed"] for row in results})
    if labels:
        matrix = confusion_matrix([row["expected"] for row in results],
                                  [row["observed"] for row in results], labels=labels)
        figure, axis = plt.subplots(figsize=(max(6, len(labels)), max(6, len(labels))))
        ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=axis, xticks_rotation=45)
        figure.tight_layout()
        figure.savefig(folder / "confusion_matrix.png")
        plt.close(figure)


def test_model(workspace, video=None, video_label=None):
    workspace = Path(workspace)
    model_path = workspace / "trained_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Train the model first: py -3 run.py --train")
    model = joblib.load(model_path)  # Only load the model that this local program created.

    validation = predictions(model, "validation")
    comparisons = [error_counts(validation, threshold) for threshold in THRESHOLDS]
    # Pick the threshold with the fewest validation mistakes. Never tune on test photos.
    chosen = min(comparisons, key=lambda row: (row["false_accepts"] + row["false_rejects"] +
                                               row["wrong_person"], row["false_accepts"],
                                               row["false_rejects"], row["threshold"]))["threshold"]
    results = predictions(model, "test")
    for row in results:
        row["threshold"] = chosen
        row["observed"] = decide(row["guess"], row["similarity"], chosen)
        row["correct"] = row["observed"] == row["expected"]

    cases = eight_learning_log_tests(workspace, model, chosen, video, video_label)
    folder = workspace / "results"
    folder.mkdir(exist_ok=True)
    save_csv(folder / "thresholds.csv", comparisons)
    save_csv(folder / "test_predictions.csv", results)
    save_csv(folder / "learning_log_T01_T08.csv", cases)
    make_graphs(folder, results, comparisons)
    summary = error_counts(results, chosen)
    summary.update({"created_at": time_now(), "chosen_threshold": chosen,
                    "validation_photos": len(validation),
                    "learning_log_cases": {case["case"]: case["result"] for case in cases},
                    "source": "LFW, selected people only; not a population-wide accuracy claim",
                    "warning": "T01 is a self-match; T03/T05 are artificial edits; T08 requires real video."})
    (folder / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nEvaluation finished. These are actual test counts, not example scores:")
    print(json.dumps(summary, indent=2))
    print(f"\nCSV files and graphs saved to: {folder}")
    return summary
