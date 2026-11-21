from pathlib import Path
import json
from facecap.dataset import make_manifest


def test_disjoint_people_files_seed(tmp_path):
    root=tmp_path/'lfw'
    for person in range(9):
        folder=root/f'person_{person:02d}'
        folder.mkdir(parents=True)
        for i in range(16):
            (folder/f'img_{i:02d}.jpg').write_bytes(f'{person}:{i}'.encode())
    manifest=make_manifest(root,tmp_path/'private.json',known_count=3,
                          unknown_validation_count=3,unknown_test_count=3,seed=42)
    assert len(manifest['people'])==9
    known={name for name,p in manifest['people'].items() if p['group']=='known'}
    v={name for name,p in manifest['people'].items() if p['group']=='unknown_validation'}
    t={name for name,p in manifest['people'].items() if p['group']=='unknown_test'}
    assert known.isdisjoint(v|t) and v.isdisjoint(t)
    assert all(r['split']=='validation' for r in manifest['images'] if r['alias'] in v)
    assert all(r['split']=='test' for r in manifest['images'] if r['alias'] in t)
    for alias in known:
        splitfiles={sp:{r['file'] for r in manifest['images'] if r['alias']==alias and r['split']==sp}
                    for sp in ('train','validation','test')}
        assert all(splitfiles.values())
        assert not (splitfiles['train']&splitfiles['test'] or splitfiles['train']&splitfiles['validation'] or splitfiles['test']&splitfiles['validation'])
    assert json.loads((tmp_path/'private.json').read_text())['seed']==42
