"""Run my face recognition Capstone project with one command: py -3 run.py"""

import os
from pathlib import Path
import subprocess
import sys
import venv

PROJECT = Path(__file__).resolve().parent

# Windows can block compiled Python libraries when they are on a shared drive.
# Keep our Python installation and downloaded data on this computer instead.
if os.name == "nt":
    DATA_FOLDER = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "CapstoneFaceDetection"
else:
    DATA_FOLDER = Path.home() / ".capstone_face_detection"

PYTHON_FOLDER = DATA_FOLDER / "python_env"
PYTHON = PYTHON_FOLDER / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PACKAGES = [
    "numpy>=1.26,<3",
    "opencv-python>=4.10,<5",
    "scikit-learn>=1.4,<2",
    "joblib>=1.3,<2",
    "matplotlib>=3.7,<4",
]


def start_python(*commands):
    """Run a command in the local virtual environment (never the network drive)."""
    return subprocess.run([str(PYTHON), *commands], cwd=DATA_FOLDER, check=True)


def get_ready():
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    if not PYTHON.is_file():
        print("Creating a Python environment on your computer...", flush=True)
        venv.create(PYTHON_FOLDER, with_pip=True)

    # Check imports, not just whether pip claims installation succeeded.
    check = subprocess.run(
        [str(PYTHON), "-c", "import cv2, sklearn, numpy, matplotlib, joblib"],
        cwd=DATA_FOLDER, capture_output=True, text=True,
    )
    if check.returncode:
        print("Installing the Python libraries. This is usually needed only once.", flush=True)
        start_python("-m", "pip", "install", "--disable-pip-version-check", *PACKAGES)
        check = subprocess.run(
            [str(PYTHON), "-c", "import cv2, sklearn, numpy, matplotlib, joblib"],
            cwd=DATA_FOLDER, capture_output=True, text=True,
        )
        if check.returncode:
            raise RuntimeError(
                "The Python libraries still won't load from the local drive.\n"
                "Try: py -3 run.py --repair\n\n" + check.stderr[-1500:]
            )


def run_project(args):
    if not args or args == ["--setup-only"]:
        from detection_software.training import train
        from detection_software.testing import test_model

        train(DATA_FOLDER)
        test_model(DATA_FOLDER)
        if not args and sys.stdin.isatty():
            choice = input("\nOpen the webcam now? (y/n): ").strip().lower()
            if choice == "y":
                from detection_software.camera import start_camera
                start_camera(DATA_FOLDER)
        return

    if args[0] in ("--camera", "camera"):
        from detection_software.faces import download_models
        from detection_software.camera import start_camera
        download_models(DATA_FOLDER / "models")
        start_camera(DATA_FOLDER)
    elif args[0] in ("--test", "test"):
        from detection_software.testing import test_model
        test_model(DATA_FOLDER)
    elif args[0] in ("--train", "train"):
        from detection_software.training import train
        train(DATA_FOLDER, force=True)
    elif args[0] == "--image":
        if len(args) < 2:
            raise SystemExit('Usage: py -3 run.py --image "C:\\path\\photo.jpg"')
        from detection_software.faces import download_models
        from detection_software.camera import check_image
        download_models(DATA_FOLDER / "models")
        check_image(DATA_FOLDER, Path(args[1]))
    elif args[0] == "--video":
        if len(args) < 3:
            raise SystemExit('Usage: py -3 run.py --video "video.mp4" K01')
        from detection_software.testing import test_model
        test_model(DATA_FOLDER, Path(args[1]), args[2])
    else:
        print("Commands: py -3 run.py [--train | --test | --camera | --image PHOTO | --video VIDEO LABEL]")


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11 or newer is needed.")

    args = sys.argv[1:]
    if args and args[0] == "--inside":
        # The second Python process runs with locally installed packages.
        run_project(args[1:])
        return

    if args == ["--repair"]:
        import shutil
        if PYTHON_FOLDER.exists():
            shutil.rmtree(PYTHON_FOLDER)
        args = []

    try:
        get_ready()
        start_python(str(PROJECT / "run.py"), "--inside", *args)
    except (subprocess.CalledProcessError, RuntimeError, OSError) as error:
        print(f"\nSomething went wrong: {error}")
        print(f"\nLocal files: {DATA_FOLDER}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
