"""Run the 8 Learning Log tests and save real results and pictures."""

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from brain import Brain, DATASET, THRESHOLD
from renderer import draw

PHOTOS = DATASET / "test_photos"
EVIDENCE = DATASET / "evidence"
KINDS = {
    "T01": ("Enrollment photo", "Tony"),
    "T02": ("New daylight photo", "Tony"),
    "T03": ("60% brightness of T02", "Tony"),
    "T04": ("Side-angle photo", "Tony"),
    "T05": ("Eye cover on T02", "Tony"),
    "T06": ("Unknown person A", "Unknown"),
    "T07": ("Unknown person B", "Unknown"),
    "T08": ("Tony webcam frame", "Tony"),
}
EXTENSIONS = (".jpg", ".jpeg", ".png")


def photo_named(folder, name):
    """Accept JPG, JPEG or PNG, regardless of capital letters."""
    if not folder.exists():
        return None
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.stem.lower() == name.lower() and path.suffix.lower() in EXTENSIONS:
            return path
    return None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    PHOTOS.mkdir(parents=True, exist_ok=True)
    person_folder = DATASET / "people" / "Tony"
    enrollment_photos = (sorted(p for p in person_folder.iterdir()
                         if p.suffix.lower() in EXTENSIONS) if person_folder.exists() else [])
    if not enrollment_photos:
        raise SystemExit("First add Tony photos to dataset/people/Tony/ and run train.py")
    if not (DATASET / "known_people.json").exists():
        raise SystemExit("Run py -3.13 train.py first to enroll Tony.")

    brain = Brain()
    if not brain.people.get("Tony"):
        raise SystemExit("Tony is not enrolled. Run py -3.13 train.py again.")

    # T01 uses the first valid enrollment photo of Tony automatically.
    first = next((p for p in enrollment_photos
                  if len(brain.analyze(cv2.imread(str(p)))) == 1), None)
    if first is None:
        raise SystemExit("Could not find a usable enrollment photo in dataset/people/Tony/")

    sources = {"T01": first}
    for test_id in ("T02", "T04", "T06", "T07", "T08"):
        sources[test_id] = photo_named(PHOTOS, test_id)

    # T03 and T05 are controlled changes to T02, not independent photos.
    daylight = cv2.imread(str(sources["T02"])) if sources["T02"] else None
    if daylight is not None:
        dim = (daylight.astype(np.float32) * 0.60).astype(np.uint8)
        sources["T03"] = EVIDENCE / "T03_input.png"
        cv2.imwrite(str(sources["T03"]), dim)

        faces = brain.analyze(daylight)
        if len(faces) == 1:
            covered = daylight.copy()
            x, y, w, h = faces[0]["box"]
            cv2.rectangle(covered, (max(0, x + w // 10), max(0, y + h // 4)),
                          (min(covered.shape[1] - 1, x + 9 * w // 10),
                           min(covered.shape[0] - 1, y + h // 2)), (0, 0, 0), -1)
            sources["T05"] = EVIDENCE / "T05_input.png"
            cv2.imwrite(str(sources["T05"]), covered)

    rows = []
    now = datetime.now(timezone.utc).isoformat()
    enroll_hashes = {digest(p) for p in enrollment_photos}

    for test_id, (description, expected) in KINDS.items():
        path = sources.get(test_id)
        row = {"test": test_id, "condition": description, "time_utc": now,
               "source": str(path) if path else "", "sha256": digest(path) if path else "",
               "faces_found": "", "detection_confidence": "", "similarity": "",
               "threshold": THRESHOLD, "expected": expected, "observed": "Not Run",
               "result": "Not Run", "note": ""}
        image = cv2.imread(str(path)) if path else None
        if image is None:
            row["note"] = "Missing photo, or T02 had no single face for the transformation"
            rows.append(row)
            print(f"{test_id}: Not Run (missing input)")
            continue

        faces = brain.analyze(image)
        row["faces_found"] = len(faces)
        if len(faces) == 0:
            row["observed"] = "No Face"
        elif len(faces) > 1:
            row["observed"] = "Multiple Faces"
        else:
            row["observed"] = faces[0]["name"]
            row["detection_confidence"] = round(faces[0]["confidence"], 4)
            if faces[0]["similarity"] is not None:
                row["similarity"] = round(faces[0]["similarity"], 4)
        row["result"] = "Pass" if row["observed"] == expected else "Fail"

        if test_id in ("T02", "T04", "T08") and row["sha256"] in enroll_hashes:
            row["result"] = "Invalid"
            row["note"] = "Identical to an enrollment photo; not an independent test"
        if test_id == "T07" and sources["T06"] and row["sha256"] == digest(sources["T06"]):
            row["result"] = "Invalid"
            row["note"] = "T06 and T07 are the same file"
        if test_id == "T03":
            row["note"] = "Digital brightness change, 60% of T02"
        if test_id == "T05":
            row["note"] = "Digital black rectangle on approximate eye area of T02"
        if test_id == "T08" and not row["note"]:
            row["note"] = "User-supplied image; keep evidence of webcam origin separately"

        picture = cv2.copyMakeBorder(draw(image, faces), 0, 55, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(35, 35, 35))
        cv2.putText(picture, f"{test_id} | {row['result']} | Expected: {expected} | Got: {row['observed']}",
                    (8, image.shape[0] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(picture, f"Faces: {len(faces)} | Similarity: {row['similarity']} | Threshold: {THRESHOLD}",
                    (8, image.shape[0] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(EVIDENCE / f"{test_id}_result.png"), picture)
        rows.append(row)
        print(f"{test_id}: {row['result']} ({row['observed']}, similarity {row['similarity']})")

    with (EVIDENCE / "test_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} test rows and screenshots in {EVIDENCE}")
    print("Note: score is cosine similarity, not identification probability.")


if __name__ == "__main__":
    main()
