"""Run my face recognition project."""

from pathlib import Path
import sys

DATA = Path(__file__).resolve().parent / "dataset"


def main():
    choice = sys.argv[1].lower() if len(sys.argv) > 1 else "train"

    try:
        if choice == "train":
            from detection_software.training import train, test
            train(DATA)
            test(DATA)
        elif choice == "test":
            from detection_software.training import test
            test(DATA, *(sys.argv[2:4]))
        elif choice == "camera":
            from detection_software.camera import camera
            camera(DATA)
        else:
            print("Use: py -3.13 run.py [train | test | camera]")
    except ImportError as error:
        print(f"Could not load a Python library: {error}")
        print("Install libraries on your PC using the command in README.md.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
