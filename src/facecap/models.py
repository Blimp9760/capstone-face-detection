"""Fetch pinned OpenCV Zoo ONNX models; never commit model binaries."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from datetime import datetime, timezone

REVISION = '48b10fa505aeb51fbd127fc60a5c660040e823b7'
EXPECTED_SHA256 = {
    'yunet': '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
    'sface': '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79',
}
MODELS = {
    'yunet': ('face_detection_yunet', 'face_detection_yunet_2023mar.onnx', 100_000),
    'sface': ('face_recognition_sface', 'face_recognition_sface_2021dec.onnx', 1_000_000),
}


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def valid_model(path:Path,min_size:int)->bool:
    if not path.is_file() or path.stat().st_size<min_size:
        return False
    with path.open('rb') as f:
        return not f.read(128).startswith(b'version https://git-lfs')


def download_models(target:Path=Path('models'))->dict:
    target.mkdir(parents=True,exist_ok=True)
    manifest={'source_repo':'https://github.com/opencv/opencv_zoo','revision':REVISION,
              'checked_utc':datetime.now(timezone.utc).isoformat(),'models':{}}
    for key,(folder,filename,min_size) in MODELS.items():
        suffix=f'models/{folder}/{filename}'
        urls=[f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/{suffix}',
              f'https://raw.githubusercontent.com/opencv/opencv_zoo/{REVISION}/{suffix}',
              f'https://github.com/opencv/opencv_zoo/raw/{REVISION}/{suffix}',
              f'https://huggingface.co/opencv/opencv_zoo/resolve/main/{suffix}']
        path=target/filename
        source='previously cached (source unavailable from file alone)'
        if not valid_model(path,min_size) or sha256(path)!=EXPECTED_SHA256[key]:
            path.unlink(missing_ok=True)
            errors=[]
            for url in urls:
                part=path.with_suffix('.download')
                try:
                    req=urllib.request.Request(url,headers={'User-Agent':'capstone-face-recognition/0.1'})
                    with urllib.request.urlopen(req,timeout=120) as response,part.open('wb') as out:
                        while True:
                            block=response.read(1<<20)
                            if not block:break
                            out.write(block)
                    if not valid_model(part,min_size):
                        raise RuntimeError('response is too small or a Git LFS pointer')
                    if sha256(part)!=EXPECTED_SHA256[key]:
                        raise RuntimeError('model SHA-256 is not the expected pinned OpenCV Zoo file')
                    part.replace(path)
                    source=url
                    break
                except Exception as exc:
                    errors.append(f'{url}: {type(exc).__name__}: {exc}')
                finally:
                    part.unlink(missing_ok=True)
            if not valid_model(path,min_size) or sha256(path)!=EXPECTED_SHA256[key]:
                raise RuntimeError(f'Cannot download {filename}; check network access and try official Zoo links.\n'+'\n'.join(errors))
        manifest['models'][key]={'file':str(path),'sha256':sha256(path),
                                 'bytes':path.stat().st_size,'source':source,'pinned_revision':REVISION,
                                 'hash_verified':sha256(path)==EXPECTED_SHA256[key]}
    (target/'model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dir',type=Path,default=Path('models'))
    args=ap.parse_args()
    print(json.dumps(download_models(args.dir),indent=2))

if __name__=='__main__':main()
