"""Download LFW using scikit-learn; use original cached JPEGs without huge arrays.

LFW is research data, not a repository of consenting participants. This module
never uploads photos, and never assigns an identity to a webcam visitor.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import hashlib
from sklearn.datasets import fetch_lfw_people


def sha256(p: Path) -> str:
    digest = hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def download_lfw(data_home: Path = Path('data/lfw_cache')) -> Path:
    """Fetch/caches original 250px JPEGs; low-res array only verifies download.

    sklearn's full array can use gigabytes, so avoid that for the dataset.
    The original high-res JPEGs remain in sklearn's disk cache.
    """
    data_home = data_home.resolve()
    data_home.mkdir(parents=True, exist_ok=True)
    fetch_lfw_people(data_home=str(data_home), resize=0.1, color=False,
                     min_faces_per_person=0, funneled=True, slice_=None)
    folder = data_home / 'lfw_home' / 'lfw_funneled'
    if not folder.is_dir():
        raise FileNotFoundError(f'Expected scikit-learn cached LFW images in {folder}')
    return folder


def split_known_images(paths: list[Path], rng: random.Random) -> dict[str,list[Path]]:
    paths = sorted(paths)
    rng.shuffle(paths)
    n = len(paths)
    n_test = max(2, int(n * .20))
    n_val = max(2, int(n * .20))
    if n - n_test - n_val < 3:
        raise ValueError('Too few independent photos for this identity')
    return {'train': paths[:n-n_val-n_test],
            'validation': paths[n-n_val-n_test:n-n_test],
            'test': paths[n-n_test:]}


def make_manifest(root: Path, output: Path, known_count: int = 6,
                  unknown_validation_count: int = 3,
                  unknown_test_count: int = 3, min_images: int = 15,
                  max_images: int = 40, seed: int = 42) -> dict:
    """Disjoint unknown-person IDs in threshold validation and final test.

    Labels K01/UVAL01/UTEST01 anonymize report outputs. Raw LFW names and all
    source image paths live only in ignored local manifest.
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    by_person = {}
    for folder in root.iterdir():
        if folder.is_dir():
            paths = sorted(folder.glob('*.jpg'))
            if len(paths) >= min_images:
                by_person[folder.name] = paths
    required = known_count + unknown_validation_count + unknown_test_count
    if len(by_person) < required:
        raise ValueError(f'Need {required} identities with >= {min_images} JPEGs; '
                         f'found {len(by_person)}. Lower --min-images or counts.')
    counts = Counter({name: len(images) for name, images in by_person.items()})
    # Pick identities with most images, then vary tie order via seed.
    order = sorted(by_person, key=lambda n: (-counts[n], n))[:required]
    rng = random.Random(seed)
    # Stable permutation removes accidental train/test person order bias.
    rng.shuffle(order)
    known = order[:known_count]
    unknown_v = order[known_count:known_count + unknown_validation_count]
    unknown_t = order[known_count + unknown_validation_count:]
    people = {}
    images = []
    for group, members in (('known', known), ('unknown_validation', unknown_v), ('unknown_test', unknown_t)):
        for ix, person in enumerate(members, 1):
            alias = ('K' if group == 'known' else 'UV' if group == 'unknown_validation' else 'UT') + f'{ix:02d}'
            selected = by_person[person].copy()
            rng.shuffle(selected)
            selected = selected[:max_images]
            if group == 'known':
                assignments = split_known_images(selected, rng)
            else:
                assignments = {'validation' if group == 'unknown_validation' else 'test': selected}
            people[alias] = {'source_identity': person, 'group': group,
                             'available_images': len(by_person[person]),
                             'selected_images': len(selected)}
            for part, paths in assignments.items():
                for file in paths:
                    images.append({'alias': alias, 'split': part, 'source_identity': person,
                                   'file': str(file.relative_to(root)), 'sha256': sha256(file)})
    result = {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source': 'Labeled Faces in the Wild (LFW), scikit-learn fetch_lfw_people, funneled original JPEGs',
              'official_docs': 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html',
              'image_root': str(root), 'seed': seed, 'min_images': min_images,
              'max_images_per_identity': max_images,
              'people': people, 'images': images}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description='Download LFW, choose independent identities and photo splits')
    p.add_argument('--data-home', type=Path, default=Path('data/lfw_cache'))
    p.add_argument('--output', type=Path, default=Path('data/dataset_manifest.json'))
    p.add_argument('--min-images', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--known', type=int, default=6)
    p.add_argument('--unknown-val', type=int, default=3)
    p.add_argument('--unknown-test', type=int, default=3)
    args = p.parse_args()
    root = download_lfw(args.data_home)
    manifest = make_manifest(root, args.output, args.known, args.unknown_val,
                             args.unknown_test, min_images=args.min_images, seed=args.seed)
    print(f'LFW cached at {root}; selected {len(manifest["people"])} identities, '
          f'{len(manifest["images"])} independent JPEG files; local manifest at {args.output}')

if __name__ == '__main__':
    main()
