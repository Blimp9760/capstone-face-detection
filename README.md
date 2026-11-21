# Face Detection Capstone

Simple project: **run.py → brain.py → renderer.py → run.py**.

- **run.py** asks whether to use a webcam or a photo from disk. It never installs anything.
- **brain.py** takes an OpenCV image and returns face boxes, detection confidence, features, and (if trained) name and cosine similarity.
- **renderer.py** draws the brain's output on the image.
- **train.py** separately downloads LFW, trains a small SVM on SFace embeddings, and writes real test results to `dataset/`.
- **dataset/** contains downloaded AI models, LFW photos, the trained classifier and test CSVs. These files are local only and ignored by GitHub.

## Install libraries yourself (one time)

Use your normal Python installation on **C:**, not the old `.venv` stored on the network drive:

```powershell
py -3.13 -m pip install --user numpy opencv-python scikit-learn joblib
```

## Use the program

```powershell
py -3.13 run.py
```

Choose **1** for the camera or **2** for an image. For images, paste the complete Windows file path. The program displays face boxes and confidence scores. **To stop the camera, focus the VS Code terminal and press Ctrl+C.** For a still image, close the image window to return.

## Add your own known people

1. Create a folder such as `dataset/people/Alex/`.
2. Put **3–5 different photos** of Alex in it. Each photo should show only Alex's face.
3. Repeat for any other people, using a different folder name for each.
4. Run:

```powershell
py -3.13 train.py people
```

Then run `py -3.13 run.py` and choose camera or image. It will compare faces against the people you added. You do **not** have to retrain the LFW model each time. To remove a person, delete their folder and rerun `train.py people` (this rebuilds `known_people.json`).

The new `dataset/known_people.json` contains face embeddings, not pictures, but is still **sensitive biometric data**. The photo folders and embeddings stay local and are excluded from GitHub. Only add people who have agreed. Recognition depends on photo quality and lighting; matching scores are not probabilities.

## Optional: LFW training experiment

To train and measure recognition on the LFW dataset, run this separately:

```powershell
py -3.13 train.py
```

This downloads the original LFW photos and OpenCV YuNet/SFace models once, then trains a **linear SVM** on selected face embeddings. It keeps training, validation and final-test photos separate and uses different people for unknown-face testing. The outputs are `dataset/results.csv`, `dataset/thresholds.csv` and `dataset/learning_log_tests.csv`. T08 is marked Not Run because no real video was supplied. These reports contain **actual outputs only when train.py is executed**.

YuNet and SFace are already pretrained; this project trains only the SVM. A detection confidence is not an identity confidence, and a cosine similarity is not a probability. LFW photos are public research images, not proof of biometric consent. Don't upload private face images or trained embeddings to GitHub.

Models: [OpenCV Zoo](https://github.com/opencv/opencv_zoo). Dataset: [LFW](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html).
