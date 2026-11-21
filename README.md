# Face detection capstone

The main program is **run.py → brain.py → renderer.py → run.py**.

- `run.py`: choose a webcam or an image and show results.
- `brain.py`: detect faces, return boxes/confidence, and match saved people.
- `renderer.py`: draw boxes and names on the picture.
- `train.py`: add your own people; optionally train an SVM on LFW for a school experiment.
- `dataset/`: photos, downloaded models and results; **not uploaded to GitHub**.

## Install (once)

Use Python installed on your C: drive, not a `.venv` on a network share.

```powershell
py -3.13 -m pip install --user numpy opencv-python scikit-learn joblib
```

## Run

```powershell
py -3.13 run.py
```

Choose **1** for camera or **2** for an image path. For the camera, focus the terminal and press **Ctrl+C** to stop. The first run downloads YuNet and SFace model files automatically.

## Teach it someone's name

1. Create `dataset/people/Alex/` and put 3–5 different photographs of Alex there.
2. Create a separate folder for each additional person. Every photo should show **exactly one** face.
3. Run `py -3.13 train.py` and then restart `run.py`.
4. To remove someone, delete their photo folder and rerun `train.py`. This rebuilds the known-person list.

Only use photos with permission. `dataset/known_people.json` contains sensitive face features and is kept out of GitHub. The initial similarity threshold is **0.45**, which may need adjustment after testing; similarity is not a probability of identity.

## Optional LFW training experiment

```powershell
py -3.13 train.py lfw
```

This downloads the LFW dataset, uses YuNet/SFace features, trains a small SVM to recognize six selected LFW identities, validates a similarity threshold, and tests on separate photos and unknown people. Results go to `dataset/lfw_results.csv`. The trained SVM is saved as `dataset/lfw_svm.joblib` for research; it is **not** required by the live app, which recognizes your own enrolled people directly.

This minimal version does not automatically reproduce the Learning Log's eight special T01–T08 tests or create charts. Those experiments would need to be run/documented separately for the full Learning Log.

Models: [OpenCV Zoo](https://github.com/opencv/opencv_zoo). Dataset: [LFW](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html). Publicly available LFW photos do not imply biometric consent.
