"""Train only a linear SVM on PRETRAINED SFace embeddings; never train YuNet/SFace."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import cv2
import joblib
import numpy as np
import sklearn
from sklearn.svm import SVC
from .dataset import sha256
from .pipeline import FacePipeline, unit


def extract_manifest(manifest_path: Path, models_dir: Path, output: Path) -> dict:
    doc = json.loads(manifest_path.read_text(encoding='utf-8'))
    root = Path(doc['image_root'])
    pipe = FacePipeline(models_dir)
    rows = []
    embeddings = []
    failures = []
    for idx, item in enumerate(doc['images'], 1):
        f = root / item['file']
        if not f.is_file() or sha256(f) != item['sha256']:
            raise ValueError(f'Missing/altered LFW source: {f}')
        frame = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if frame is None:
            failures.append({**item, 'reason': 'input_failure: unreadable image', 'faces_detected': 0})
            continue
        emb, n, _ = pipe.extract_one(frame)
        if emb is None:
            failures.append({**item, 'reason': ('feature_failure' if n else 'detection_failure'), 'faces_detected': n})
            continue
        embeddings.append(emb)
        rows.append({'alias': item['alias'], 'split': item['split'],
                     'sha256': item['sha256'], 'source_file': item['file'], 'faces_detected': n})
        if idx % 50 == 0:
            print(f'Extracted {idx}/{len(doc["images"])} images... ', flush=True)
    if not embeddings:
        raise RuntimeError('No faces extracted; confirm OpenCV models and LFW input data')
    X = np.asarray(embeddings, dtype=np.float32)
    payload = {'X': X, 'rows': rows, 'failures': failures,
               'manifest': doc, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'versions': {'python': sys.version.split()[0], 'opencv': cv2.__version__,
                            'scikit_learn': sklearn.__version__, 'numpy': np.__version__,
                            'platform': platform.platform()},
               'model_sha256': {f.name: sha256(f) for f in models_dir.glob('*.onnx')}}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output, compress=3)
    print(f'Embedding extraction: {len(rows)} successes, {len(failures)} failures; saved locally to {output}')
    return payload


def train_svm(payload: dict, output: Path) -> dict:
    rows = payload['rows']
    train_idx = [i for i, r in enumerate(rows) if r['split'] == 'train' and r['alias'].startswith('K')]
    y = [rows[i]['alias'] for i in train_idx]
    counts = Counter(y)
    if len(counts) < 2:
        raise ValueError('Training requires >=2 known identities detected successfully')
    if min(counts.values()) < 2:
        raise ValueError(f'Training requires >=2 detected images for every class: {counts}')
    X = np.asarray([unit(payload['X'][i]) for i in train_idx], dtype=np.float32)
    clf = SVC(kernel='linear', C=1.0, random_state=payload['manifest']['seed'])
    clf.fit(X, y)
    gallery = {key: X[np.asarray(y) == key] for key in sorted(counts)}
    artifact = {'classifier': clf, 'gallery': gallery,
                'classes': sorted(counts), 'train_counts': dict(counts),
                'trained_utc': datetime.now(timezone.utc).isoformat(),
                'seed': payload['manifest']['seed']}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    print(f'Trained linear SVM on {len(X)} SFace vectors across {len(counts)} known identities -> {output}')
    return artifact


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=Path('data/dataset_manifest.json'))
    p.add_argument('--models', type=Path, default=Path('models'))
    p.add_argument('--embeddings', type=Path, default=Path('local/lfw_embeddings.joblib'))
    p.add_argument('--classifier', type=Path, default=Path('local/lfw_svm.joblib'))
    args = p.parse_args()
    payload = extract_manifest(args.manifest, args.models, args.embeddings)
    train_svm(payload, args.classifier)

if __name__ == '__main__':
    main()
