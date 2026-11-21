# Test plan and failure analysis

## What was tested here vs what needs local execution

- **Offline source tests:** `PYTHONPATH=src python -m pytest -q`; no network, camera, pretrained model or personal face data required.
- **Real SFace and LFW experiments:** `python run.py --setup-only`; downloads raw JPEGs and ONNX models onto the runner's machine. **Not executed** in an environment with no outbound DNS/network. Do not quote artificial scores as outcomes.
- **Live camera:** `python run.py --camera` on a computer with local webcam permission; enroll only with consent. **Not verified here**.
- **Prerecorded video:** must be separately supplied with an expected label; **T08 = Not Run** otherwise.

## Stages and what errors mean

1. **Image load**: unreadable file or frame → input failure, not a biometric rejection.
2. **YuNet face detection**: no face boxes → detection failure, not a similarity score.
3. **SFace alignment/feature**: a detected face whose alignment/feature step fails → separately counted `feature_failure`, not Unknown.
4. **Trained SVM**: selects the most likely *among its training classes* even for a stranger; it is **closed-set** by itself.
5. **Cosine check**: reject chosen class if its closest enrolled training example is below threshold. Labels are `Known class` or `Unknown`.
6. **Final outcome**: count wrong-identity predictions, false accepts and false rejects separately. Detection failures never silently become true Unknown results.

## Denominators

- `total_images`: every held-out test source image, including no-face cases.
- `faces_scored`: those yielding a usable feature for the largest detected face.
- `known_test_images`: all held-out source files from known identities.
- `unknown_test_images`: all files from the disjoint unknown-test identities.
- `false_accepts`: an unknown person accepted as a known class.
- `false_rejects`: a known person scored but rejected as Unknown.
- `wrong_identities`: a known person scored as the wrong known class.
- `detection_failures`: images where YuNet reported no detected face.
- `feature_failures`: a detected face whose alignment/embedding extraction failed.
- `input_failures`: unreadable input files.
- `accuracy_on_images`: correct known or correct unknown labels divided by total source images. **A convenient descriptive count, not population accuracy**.

SVM uses SFace embeddings that were extracted from train images, and the gallery includes those **training** embeddings only. Known validation and test images are distinct JPEGs. Unknown validation and unknown test **identities** never overlap, and no unknown identities are used to train the SVM. The T01 self-match is included for program sanity but not in held-out accuracy.

## Test run record checklist

| Evidence | Where to find it |
|---|---|
| Actual stage timestamps, failures and setup success | `reports/run_audit.json` |
| Model provenance and SHA-256 hashes | `models/model_manifest.json` + `reports/metrics.json` |
| Input files, original person names, splitting and SHA-256 | `data/dataset_manifest.json` (private) |
| Real run timestamp and package versions | `reports/metrics.json` |
| Each final test prediction and stage failure | `reports/test_rows.csv` |
| Eight learning log cases, transform and video status | `reports/learning_log_T01_T08.csv` |
| Threshold comparisons | `reports/threshold_comparison.csv`, `reports/validation_thresholds.png` |
| Class-level test errors | `reports/test_confusion_matrix.png` |
| Local enrollment/restart/delete tests | `tests/test_store.py` and `pytest` output |

## Eight cases: careful interpretations

- **T01** self-match generally has score near 1; it cannot test generalization.
- **T02** independent file is stronger than an alternate crop, but LFW may include near-duplicate photographs.
- **T03** digital darkening simulates a pixel transformation, not a dim physical room.
- **T04** a separate photograph is not automatically a verified different head angle.
- **T05** masking eyes is synthetic; no-face is a **detection** failure. It is not an SFace low-score case.
- **T06/T07** use **two different unknown identities**, not crops of the same other person.
- **T08** must be an actual decoded video frame, *not a still proxy*. Without an eligible video and expected label, it is Not Run.

## Local webcam/checklist

1. Consent to enrollment and record the exact retention plan outside the repository.
2. Start webcam (`facecap camera`) and confirm "LOCAL PROCESSING" label.
3. Press `E`, confirm permission and enroll a person; repeat with a second allowed photo or view.
4. Move to new lighting/pose, note displayed name and similarity, including failures.
5. Press `D`, type exact name, type `DELETE`, quit and **restart** app, press `L`; confirm name is absent.
6. Confirm returning person is now Unknown at the same threshold. Deletion does not securely erase snapshots/backups or LFW research data.

## Limitations

Selected, modest LFW identity groups and the people/photos in them do not represent the population. Results depend on which JPEGs successfully yield faces, and there may be duplicates, demographic biases and pose differences. The selected threshold may not transfer to a live webcam. A rigorous follow-up should use independently acquired, consented images, multiple physical lighting/pose conditions and evaluate error rates with clearly stated denominators and participants' permissions.
