"""Convenient Python command-line entrypoint."""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timezone
import json
from . import __version__
from .models import download_models
from .dataset import download_lfw, make_manifest
from .train import extract_manifest, train_svm
from .evaluate import evaluate
from .app import webcam, recognize_image
from .store import EnrollmentStore


def main():
    p=argparse.ArgumentParser(description='Local face recognition capstone')
    p.add_argument('--version',action='version',version=__version__)
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('setup',help='Download data/models, train and evaluate (first run)')
    a=sub.add_parser('camera',help='Local webcam recognition and enrollment')
    a.add_argument('--threshold',type=float,default=None)
    a.add_argument('--index',type=int,default=0)
    a=sub.add_parser('image',help='Recognize saved image against locally enrolled identities')
    a.add_argument('path',type=Path)
    a.add_argument('--threshold',type=float,default=None)
    sub.add_parser('list',help='List locally enrolled identities')
    a=sub.add_parser('delete',help='Delete a locally enrolled identity')
    a.add_argument('identity')
    args=p.parse_args()
    if args.cmd=='setup':
        audit={'started_utc':datetime.now(timezone.utc).isoformat(),
               'status':'running', 'steps':[]}
        Path('reports').mkdir(exist_ok=True)
        def audit_save():
            (Path('reports')/'run_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
        def step(label,fn):
            record={'stage':label,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
            audit['steps'].append(record);audit_save()
            print(f'\n=== {label} ===',flush=True)
            try:
                result=fn()
                record['status']='success'
                return result
            except Exception as ex:
                record['status']='failed'
                record['error']=f'{type(ex).__name__}: {str(ex)[:1000]}'
                audit['status']='failed'
                raise
            finally:
                record['finished_utc']=datetime.now(timezone.utc).isoformat()
                audit_save()
        try:
            step('download verified OpenCV model files',lambda: download_models())
            root=step('download original LFW JPEGs',lambda: download_lfw())
            step('select LFW identities and split photo files',lambda: make_manifest(root,Path('data/dataset_manifest.json')))
            payload=step('extract YuNet/SFace embeddings',lambda: extract_manifest(Path('data/dataset_manifest.json'),Path('models'),Path('local/lfw_embeddings.joblib')))
            step('fit actual linear SVM',lambda: train_svm(payload,Path('local/lfw_svm.joblib')))
            step('validation, threshold selection and held-out evaluation',lambda: evaluate(Path('local/lfw_embeddings.joblib'),Path('local/lfw_svm.joblib'),Path('models'),Path('reports')))
        except Exception:
            audit['status']='failed'
            audit_save()
            raise
        audit['status']='completed';audit['finished_utc']=datetime.now(timezone.utc).isoformat();audit_save()
        print('SETUP COMPLETE. Review reports/run_audit.json, metrics.json and learning_log_T01_T08.csv.')
    elif args.cmd in ('camera','image'):
        threshold=args.threshold
        if threshold is None:
            # The LFW-tuned experiment threshold may be too strict/lenient for a new person.
            import json
            metrics=Path('reports/metrics.json')
            threshold=json.loads(metrics.read_text())['selected_threshold'] if metrics.is_file() else .363
        if not -1<=threshold<=1: p.error('threshold must be between -1 and +1')
        if args.cmd=='camera':webcam(Path('models'),Path('local/enrollments.json'),threshold,args.index)
        else:recognize_image(args.path,Path('models'),Path('local/enrollments.json'),threshold)
    elif args.cmd=='list':print('\n'.join(EnrollmentStore().list_names()) or '(none)')
    elif args.cmd=='delete':
        if input(f'Type DELETE to remove {args.identity}: ').strip()=='DELETE':
            print('Deleted' if EnrollmentStore().delete(args.identity) else 'Not found')
        else:print('Cancelled')

if __name__=='__main__':main()
