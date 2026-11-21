"""Webcam demo. Press E to save a face and D to delete one."""

import json
from pathlib import Path
import cv2

from detection_software.faces import FaceFinder, similarity

THRESHOLD = 0.363


def camera(data):
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    saved_file = data / "saved_faces.json"
    people = json.loads(saved_file.read_text(encoding="utf-8")) if saved_file.exists() else {}
    finder = FaceFinder(data)
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise RuntimeError("The webcam could not be opened")

    print("E = enroll | D = delete | Q = quit")
    try:
        while True:
            ok, image = webcam.read()
            if not ok:
                break
            faces = finder.find(image)
            for face in faces:
                name, best = "Unknown", -1.0
                for person, saved_vectors in people.items():
                    for vector in saved_vectors:
                        score = similarity(face["vector"], vector)
                        if score > best:
                            name, best = person, score
                if best < THRESHOLD:
                    name = "Unknown"

                x, y, w, h = face["box"]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(image, f"{name} ({best:.2f})", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            cv2.putText(image, f"LOCAL | threshold {THRESHOLD} | E: save  D: delete  Q: quit",
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
            cv2.imshow("Face Recognition", image)
            key = cv2.waitKey(1) & 255
            if key in (ord("q"), 27):
                break
            if key == ord("e") and faces:
                # Save one face, never the camera photograph.
                face = max(faces, key=lambda x: x["box"][2] * x["box"][3])
                name = input("Person's name: ").strip()
                if name and input("Do they agree to save their face features? (yes/no): ").lower() == "yes":
                    people.setdefault(name, []).append(face["vector"].tolist())
                    saved_file.write_text(json.dumps(people), encoding="utf-8")
                    print(f"Saved {name} locally")
            if key == ord("d"):
                name = input("Name to delete: ").strip()
                if name in people and input(f"Type DELETE to remove {name}: ") == "DELETE":
                    del people[name]
                    saved_file.write_text(json.dumps(people), encoding="utf-8")
                    print(f"Deleted {name}")
    finally:
        webcam.release()
        cv2.destroyAllWindows()
