# Learning Log 2 — actual implementation and evidence map

**Essential question:** How can I create a program that finds a face in a photo or webcam video and tells whether it has seen that person before?

**Evidence rule:** This is a record of *source code and available verified tests*, NOT a replacement for authentic student reflections, participant consent records, teacher feedback or actual time spent. The attached Learning Log's example values (0.921, 0.774, 0.681, 0.812 etc.) were **illustrative** and never used as measured outcomes.

## Completed technical implementation

| Log requirement | Implemented source | How to run / check | Actual verification status |
|---|---|---|---|
| Detection → alignment → features | `src/facecap/pipeline.py` | `python -m facecap.models`, `facecap camera` | Source written. Real YuNet/SFace weights not available in build container. |
| LFW download and distinct identity sets | `src/facecap/dataset.py` | `python -m facecap.dataset` | Deterministic synthetic-folder split checked by offline unit test; real LFW download pending network-accessible device. |
| SVM supervised training | `src/facecap/train.py` | `python -m facecap.train` | Source written; real SVM fit on LFW awaits model download and embedding extraction. |
| Threshold evaluation on validation only | `src/facecap/recognition.py`, `src/facecap/evaluate.py` | `python -m facecap.evaluate` | Pure decision logic verified by offline unit tests; LFW metrics pending local execution. |
| Real T01–T08 test register | `src/facecap/evaluate.py`, `docs/TESTING.md` | Look at `reports/learning_log_T01_T08.csv` after run | Conditions implemented, not falsely claimed executed. T08 is Not Run unless a genuine video supplied. |
| Local UI with label, similarity, threshold | `src/facecap/app.py` | `python -m facecap.cli camera` | Source written; live camera requires desktop hardware and consent, not verified here. |
| Enrollment, persistence, deletion and restart | `src/facecap/store.py`, `tests/test_store.py` | `python -m pytest -q` | Offline persisted/reloaded/deleted JSON store tested using synthetic vectors. Does not prove secure disk erasure. |
| Anonymized report and plots | `src/facecap/evaluate.py` | `reports/metrics.json` and two PNG plots | Implemented; numeric results not available before real dataset run. |
| Privacy and documented provenance | `.gitignore`, `docs/ETHICS.md`, `docs/SOURCES.md` | inspect local manifests/hashes | Code files exist; model hashes and input identities will be written only on local successful execution. |

## Builder environment test record

- Python source compiled with `python -m compileall -q src run.py`.
- Automated tests run with `PYTHONPATH=src python -m pytest -q`.
- **Observed result:** `11 passed` (the initial run exposed one failure which was corrected), as measured on the build environment. The corrected bug was a NumPy array truth-value check in the SVM gallery gate: `if not candidates` failed for an ndarray; updated to `if len(candidates) == 0` and reran successfully.
- Network download attempt: requesting pinned YuNet ONNX binary at `raw.githubusercontent.com` failed because outbound DNS/network access was unavailable in the build container. No real download, SFace feature extraction, SVM training or webcam experiment was asserted.
- Build-session calendar date: November 21, 2026. (Exact machine-local execution times for the future full experiment are recorded by `metrics.json` and case CSV.)

## What should be added **after** a successful run on your computer

1. Link to the GitHub commit SHA and a trusted terminal transcript of your `python run.py --setup-only` run.
2. Keep `data/dataset_manifest.json` and `models/model_manifest.json` privately so source-image and model SHA-256 hashes can be audited. Do **not** commit those raw identity records.
3. Describe raw `reports/test_rows.csv`, `reports/threshold_comparison.csv` and `reports/learning_log_T01_T08.csv` values, with the denominators and an error/failure analysis in your own words.
4. If you actually perform the webcam, lighting/pose, user-feedback and delete/restart activities, record permission, dates, screenshots, predictions, surprising outcomes and what you changed. Mark all unperformed activities as **Not Run**.
5. Add verified AI-use links and teacher feedback only if you actually have original conversation/source records; do not invent them.
6. After reviewing real results, write your personal reflection explaining one prediction, one failed test, a revised decision and what you'd do next. This file does not impersonate student reflection.

## Planned next measurements (not completed)

- Several genuinely independent, consented live camera photos across multiple lighting and poses; avoid claiming that digital darkening is physical darkness.
- At least two distinctly unfamiliar people and repeat trials, with explicit false-match/false-reject counts and no high-stakes use.
- Camera enrollment/deletion/restart persistence and a clear explanation of what file deletion cannot erase.
- A real prerecorded video test for T08 if a permitted video becomes available.

**Interpretation:** The authentic evidence so far establishes implementation and runnable offline unit tests, not face-recognition accuracy on real people. Future outcomes should be reported from local files produced by the Python programs, not reconstructed from the original illustrative table.
