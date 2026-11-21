# Publishing and syncing this project

Connected repository: https://github.com/Blimp9760/capstone-face-detection

Source code and documentation were published through the connected GitHub integration; raw images, the LFW dataset, ONNX weights, trained embeddings, and local biometric records were not uploaded. The repository was **public at initial publication**. To change this, open GitHub repository Settings > General > Danger Zone > Change repository visibility (GitHub's wording can vary).

## Use GitHub Desktop on your computer

1. Open GitHub Desktop > File > Clone repository, select `Blimp9760/capstone-face-detection` and choose a local folder.
2. Run `start_windows.bat` on Windows or `python3 run.py` on macOS. The first run downloads models and LFW files locally and runs evaluation.
3. Review GitHub Desktop Changes before committing; `.gitignore` excludes face images, LFW data, models, personal embeddings, and generated reports.
4. Never use `git add -f` to include face images or biometric data. Only commit reviewed source changes and non-sensitive documentation.

After a run, inspect `reports/metrics.json`, `reports/test_rows.csv`, `reports/learning_log_T01_T08.csv`, and `reports/run_audit.json` on your own computer. They are local-only until you explicitly review and share them.
