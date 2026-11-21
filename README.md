# My Face Recognition Project

A Python project that finds faces with YuNet, extracts features with SFace, and trains an SVM using the LFW photo dataset. The webcam can also remember and delete people who agree to be enrolled.

## Install (one time)

Use **Python 3.13 installed on your C: drive**, not the old network-drive `.venv`. Open PowerShell:

```powershell
py -3.13 -m ensurepip --upgrade
py -3.13 -m pip install --user numpy opencv-python scikit-learn joblib
```

If `pip` already works, skip the `ensurepip` line. No virtual environment is needed. In VS Code, choose your regular Python 3.13 interpreter, not `.venv`.

## Run

```powershell
py -3.13 run.py           # Download LFW/models, train, then test
py -3.13 run.py camera    # Open webcam (E = enroll, D = delete, Q = quit)
py -3.13 run.py test      # Repeat the tests
```

The first run downloads the models and LFW photographs into `dataset/`. They are not uploaded to GitHub. Test results are also saved there. Later runs reuse the saved model. To retrain from scratch, delete `dataset/trained.joblib` and run again.

## Files

- `run.py` — Starts the program. Nothing is installed automatically.
- `detection_software/faces.py` — Detects faces and extracts their features.
- `detection_software/training.py` — Downloads photos, trains the SVM and runs the tests.
- `detection_software/camera.py` — Webcam and save/delete controls.
- `dataset/` — Downloaded photos, model files, results and saved faces (private; ignored by GitHub).

The Learning Log tests T01–T07 run with LFW photographs. T08 needs a real video (`py -3.13 run.py test video.mp4 Unknown`) and otherwise says **Not Run**. T03 and T05 change an image digitally, not a real lighting or occlusion experiment. The pretrained YuNet/SFace networks are not retrained; only the SVM is trained here. A similarity score is **not** a probability. Only enroll people who consent.

Model sources: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface). Dataset: [LFW](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html). Public photos do not imply consent for personal identification use.
