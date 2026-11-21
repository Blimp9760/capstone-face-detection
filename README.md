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

## Learning Log test evidence

Put five new photographs inside `dataset/test_photos/`:

- `T02.jpg` — Tony, daylight, **not** used for enrollment.
- `T04.jpg` — Tony, side angle, a different photograph.
- `T06.jpg` — One person who has **not** been enrolled.
- `T07.jpg` — A second, different person who has **not** been enrolled.
- `T08.jpg` — An actual webcam frame showing Tony, separate from enrollment photos.

JPG, JPEG and PNG are accepted (keep the same `T02`, `T04`, etc. filename stems). T01 automatically uses a photo from `dataset/people/Tony/`. The script creates T03 by darkening T02 to 60% and T05 by drawing a black rectangle over approximately the eye region of T02. They are **digital transformations**, not independently captured conditions. Each supplied photo should contain exactly one visible face, and the two unknown people should have consented.

Run `py -3.13 train.py` to enroll Tony first, then:

```powershell
py -3.13 test.py
```

Open `dataset/evidence/test_results.csv` and `T01_result.png` through `T08_result.png`. It also saves `T03_input.png` and `T05_input.png`. Missing inputs are marked **Not Run** and exact duplicates of enrollment images in T02/T04/T08 are marked **Invalid**. Do not replace failed results with expected results. The T08 input is user-supplied, so keep an independent screenshot proving it came from the webcam. The script reports direct-comparison performance at the current `brain.py` threshold of 0.45, not the optional LFW SVM experiment.

Models: [OpenCV Zoo](https://github.com/opencv/opencv_zoo). Dataset: [LFW](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html). Publicly available LFW photos do not imply biometric consent.