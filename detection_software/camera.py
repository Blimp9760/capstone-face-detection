"""Use the webcam and optionally remember or delete someone locally."""

from pathlib import Path
import cv2

from detection_software.faces import FaceDetector, SavedFaces


def threshold_from_tests(workspace):
    # The research threshold is only a starting point for webcam users.
    import json
    file = Path(workspace) / "results" / "results.json"
    if file.exists():
        return json.loads(file.read_text(encoding="utf-8"))["chosen_threshold"]
    return 0.363


def check_image(workspace, file):
    detector = FaceDetector(Path(workspace) / "models")
    image = cv2.imread(str(file))
    if image is None:
        raise ValueError(f"Cannot read image: {file}")
    saved = SavedFaces(Path(workspace) / "saved_faces.json")
    threshold = threshold_from_tests(workspace)
    for face in detector.find_faces(image):
        name, score = saved.recognize(face["features"], threshold)
        print(f"{name} | similarity: {score} | threshold: {threshold}")
    if detector.faces_found == 0:
        print("No face detected")


def start_camera(workspace):
    workspace = Path(workspace)
    detector = FaceDetector(workspace / "models")
    saved = SavedFaces(workspace / "saved_faces.json")
    threshold = threshold_from_tests(workspace)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Webcam could not be opened. Check camera access in Windows settings.")

    print("\nWebcam commands: E = enroll, D = delete, L = list people, Q = quit")
    print("Obtain permission before enrolling anyone. Face features stay on your computer.")
    try:
        while True:
            ok, image = camera.read()
            if not ok:
                print("Could not read webcam frame")
                break
            faces = detector.find_faces(image)
            for face in faces:
                name, score = saved.recognize(face["features"], threshold)
                x, y, width, height = face["box"]
                color = (0, 200, 0) if name != "Unknown" else (0, 150, 255)
                cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
                details = f"{name} | {score:.3f}" if score is not None else "Unknown"
                cv2.putText(image, details, (max(0, x), max(18, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.rectangle(image, (0, 0), (image.shape[1], 63), (30, 30, 30), -1)
            cv2.putText(image, f"LOCAL ONLY | threshold {threshold:.3f}", (12, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            cv2.putText(image, "E: enroll  D: delete  L: list  Q: quit", (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)
            cv2.imshow("My Face Recognition Project", image)
            key = cv2.waitKey(1) & 255
            if key in (ord("q"), 27):
                break
            if key == ord("l"):
                print("Saved names:", ", ".join(saved.people) or "(none)")
            if key == ord("e"):
                if not faces:
                    print("No face to enroll")
                    continue
                biggest = max(faces, key=lambda face: face["box"][2] * face["box"][3])
                name = input("Name to enroll: ").strip()
                if name and input(f"Does {name} agree to save their face features locally? (yes/no): ").strip().lower() == "yes":
                    saved.add(name, biggest["features"])
                    print("Saved locally (no photo saved).")
                else:
                    print("Cancelled")
            if key == ord("d"):
                name = input("Name to delete: ").strip()
                if input(f"Type DELETE to delete {name}: ").strip() == "DELETE":
                    print("Deleted" if saved.delete(name) else "Name not found")
    finally:
        camera.release()
        cv2.destroyAllWindows()
