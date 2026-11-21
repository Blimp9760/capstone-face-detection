# Troubleshooting

### Python isn't found

Install Python 3.11+ from https://www.python.org/downloads/, check “Add python.exe to PATH” on Windows, open a *new* terminal and retry. Use `py -3 run.py` on Windows, `python3 run.py` on macOS.

### Dependencies fail to install

First run requires internet. Antivirus or workplace/school network filters sometimes block the Python package index; use a permitted connection. Remove `.venv/` and rerun if the environment was only partly created. On macOS you may need Apple command-line developer tools for certain source builds; recent supported Python releases generally have prebuilt wheels.

### The OpenCV model download fails

Some GitHub responses return a **Git LFS pointer** instead of an ONNX binary. `facecap.models` tries three URLs pinned to the same OpenCV Zoo revision and rejects too-small/LFS-pointer content. You can download the exact filenames from OpenCV Zoo model directories and place them in `models/` manually, then run `python run.py --setup-only` again. Check `models/model_manifest.json` for hashes after success. Ensure the filenames are `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`.

### LFW dataset download fails

Use a machine/connection that allows scikit-learn's documented dataset mirrors. Try `.venv/bin/python -m facecap.dataset` (on Windows `.venv\Scripts\python.exe -m facecap.dataset`) to isolate the error. It may require substantial disk space during downloads. No dataset is bundled with the GitHub ZIP.

### Too few images or training classes

LFW file counts or detector failures may reduce usable training examples. Run `python -m facecap.dataset --min-images 10` if necessary; check `data/dataset_manifest.json`. If too many photos fail at the detection/feature stage, try an alternate image source with permission instead of inventing matches. This may change experiment design and must be documented.

### Camera shows black or can't open

Grant Python/Terminal/PowerShell camera permission in your OS settings, close other camera software and try `python run.py --camera`. The system running in a server or notebook has no webcam; use a personal desktop/laptop.

### Similarity scores look low / most faces show Unknown

Try a properly consented enrollment with several varied photos/frames, but recognize that higher thresholds naturally reject more samples. `.363` is **a demo reference**, and your local experimental threshold may be different. Do not label cosine score as `confidence %`. A high SVM research-test accuracy does not imply strong live webcam performance.

### `No Face` but the person was visible

This is a **detection failure**, not necessarily proof of a different identity. Review lighting, image blur, face angle and occlusion; count these separately from wrong labels or threshold rejections.

### Saving/distributing results

`reports/` is ignored by default. To include an anonymized result, inspect that file and then explicitly override the ignore (`git add -f reports/safe_summary.csv`) only if it is safe. Never add face photos, embedding arrays, saved enrollment records, dataset directories, or unrestricted biometric artifacts.
