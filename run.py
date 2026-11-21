"""Choose a picture or camera, then show the brain's results."""

import cv2
from brain import Brain
from renderer import draw


def main():
    print("1 - Camera")
    print("2 - Image")
    choice = input("Choose: ").strip()
    if choice not in ("1", "2"):
        print("Please choose 1 or 2.")
        return

    brain = Brain()

    if choice == "2":
        path = input("Image path: ").strip().strip('"')
        image = cv2.imread(path)
        if image is None:
            print("Could not open that image.")
            return
        faces = brain.analyze(image)
        for face in faces:
            print({k: v for k, v in face.items() if k != "features"})
        cv2.imshow("Faces", draw(image, faces))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open the camera.")
        return
    print("Camera running. Focus this terminal and press Ctrl+C to stop.")
    try:
        while True:
            ok, image = camera.read()
            if not ok:
                break
            faces = brain.analyze(image)
            cv2.imshow("Faces", draw(image, faces))
            cv2.waitKey(1)
            if cv2.getWindowProperty("Faces", cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
