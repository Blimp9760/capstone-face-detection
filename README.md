# My Face Recognition Capstone

**Question:** Can I make a Python program that finds a face in a photo or webcam video and decides if it has seen that person before?

## Run it

Install Python 3.11 or newer. On Windows, open this folder in VS Code or PowerShell and type:

```powershell
py -3 run.py
```

On macOS, use `python3 run.py`.

The first time, the script installs what it needs, downloads the **Labeled Faces in the Wild (LFW)** research dataset and the **YuNet/SFace** OpenCV models, trains a small SVM classifier, and tests its predictions. It then asks if you want to try the webcam.

**Fix for network-drive errors:** Your Python libraries, models, dataset, and reports are now saved on your **local computer**, even if the project itself is on a mapped drive like `I:` or a `\\server\share` path. On Windows, everything downloaded or generated goes to:

`%LOCALAPPDATA%\CapstoneFaceDetection`

Your previous `.venv` on the network drive is no longer used. If the local Python environment needs rebuilding, run `py -3 run.py --repair`.

## Other commands

```powershell
py -3 run.py --camera     # Webcam, with E to enroll, D to delete, Q to quit
py -3 run.py --train      # Download data and retrain the classifier
py -3 run.py --test       # Repeat the saved evaluation
py -3 run.py --image "C:\path\photo.jpg"
py -3 run.py --video "C:\path\recording.mp4" K01
```

When testing a video, change `K01` to the expected research identity or `Unknown`.

## What the files do

- `run.py` — The only script you need to start the project. It also fixes the network-drive installation problem.
- `detection_software/faces.py` — Downloads YuNet/SFace, detects faces, extracts feature vectors and compares them.
- `detection_software/training.py` — Downloads LFW, divides photos into training/validation/testing and trains an SVM on the features.
- `detection_software/testing.py` — Compares thresholds and saves real test results, including T01–T08 from my Learning Log.
- `detection_software/camera.py` — The webcam and enrollment/deletion controls.
- `dataset/` — Placeholder for dataset files. Downloads are saved on the local computer so the GitHub repository stays small and does not publish face photos.
- `.gitignore` — Protects downloaded photos, models and personal face data from accidentally being committed.

## How it works

1. **YuNet** finds a face and its landmarks.
2. **SFace** aligns it and converts the face into a feature vector (numbers).
3. **An SVM trained on LFW** guesses which research identity it looks like.
4. **Cosine similarity** checks the guess. If the score is below the chosen threshold, the answer is **Unknown**.
5. The webcam uses a **separate local enrollment list**, rather than retraining the SVM for each visitor.

The actual results will be in the local `results/` folder: `results.json`, `test_predictions.csv`, `thresholds.csv`, `learning_log_T01_T08.csv`, and two graphs. **T08 is Not Run unless a real video is supplied.** If an experiment fails or cannot run, the report should say so. Cosine similarity is not a confidence percentage.

LFW images are public research data, **not proof of biometric consent**. Only enroll webcam participants who agree. Deletion removes records from the saved enrollment file but is not secure erasure of every possible backup. The results describe this small selected dataset, not universal face-recognition accuracy. Do not upload original photos, embeddings or personal face data to GitHub.

Sources: [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface), [OpenCV tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html), [LFW dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html).

> The code is AI-assisted. Actual experimental findings and any personal reflection should be checked and recorded after running the program yourself.
