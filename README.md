# My Face Recognition Project

I wrote this Python project to detect faces in pictures and webcam video and check whether a face has been seen before.

## What the files do

- **run.py** — The main program: camera, download LFW, train an SVM, and run experiments.
- **brain.py** — The part that finds faces. Give it an OpenCV image and it returns a bounding box, detection confidence, and face features. It can also compare saved faces.
- **dataset/** — Downloads, trained model, test CSV files and locally enrolled faces. Nothing in this folder is uploaded to GitHub.

YuNet finds faces; SFace converts faces into feature vectors. I train an SVM on features from the [LFW dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html). The camera instead compares against locally enrolled face vectors, so it can remember someone without retraining the SVM.

## Install once (Windows PowerShell)

Use your normal Python installation, not a `.venv` on a shared/network drive:

```powershell
py -3.13 -m pip install numpy opencv-python scikit-learn joblib
```

If pip is missing from that **normal** Python installation, run `py -3.13 -m ensurepip --upgrade` first. Installation is manual; running the project doesn't install anything.

## Run

```powershell
py -3.13 run.py             # Easy numbered menu
py -3.13 run.py camera      # Webcam: E save, D delete, Q quit
py -3.13 run.py train       # Download LFW and train the SVM
py -3.13 run.py test        # Test results saved as CSV in dataset/
py -3.13 run.py test clip.mp4 Unknown  # Optional T08 real video
```

Training and models download only when you choose a function that needs them. The first training run may download a large photo dataset. To retrain, delete `dataset/trained.joblib` and run the train command again.

## What the numbers mean

`box` is `[left, top, width, height]` in pixels. `confidence` says how sure **YuNet is that it detected a face**, not who it is. `similarity` is cosine similarity between face features, **not a probability** that the name is correct.

The eight Learning Log test cases are saved to `dataset/learning_log_tests.csv`. T01 is a self-match, T03/T05 are artificial image edits, T04 is a separate photo (pose not verified), and T08 is Not Run unless you provide a real video. This is a classroom demonstration, not a security system. Only store people's features with permission. Public dataset availability does not mean biometric consent.

OpenCV model sources: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).
