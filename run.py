"""Choose camera or image, get results from the brain, and display them."""

from pathlib import Path
import cv2
from brain import Brain
from renderer import draw

print("1 - Camera")
print("2 - Image from disk")
choice = input("Choose 1 or 2: ").strip()

if choice == "2":
    path = input("Full image path: ").strip().strip('"').strip("'")
    image = cv2.imread(path)
    if image is None:
        raise SystemExit("no image neegy")
elif choice == "1":
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Could not open the camera.")
else:
    raise SystemExit("tf u tryna press")

try:
    brain = Brain(Path(__file__).resolve().parent / "dataset")

    if choice == "2":
        faces = brain.analyze(image)
        for face in faces:
            print({key: value for key, value in face.items() if key != "features"})
        if not faces:
            print("No faces found.")
        cv2.imshow("Face Detection", draw(image, faces))
        while cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(100) != -1:
                break
    else:
        print("Camera is running.")
        while True:
            okay, image = camera.read()
            if not okay:
                break
            cv2.imshow("Face Detection", draw(image, brain.analyze(image)))
            cv2.waitKey(1)
            if cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
except KeyboardInterrupt:
    print("Stopped.")
finally:
    if choice == "1":
        camera.release()
    cv2.destroyAllWindows()
