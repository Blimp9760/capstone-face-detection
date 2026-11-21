"""Open-set validation/test, eight Learning Log cases, counts and plots."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from .pipeline import FacePipeline
from .recognition import THRESHOLDS, predict_open_set, select_threshold
from .dataset import sha256


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def calculate_rows(payload: dict, trained: dict, split: str) -> list[dict]:
    result = []
    for ix, r in enumerate(payload['rows']):
        if r['split'] != split:
            continue
        expected = r['alias'] if r['alias'].startswith('K') else 'Unknown'
        label, score, svm_label = predict_open_set(trained['classifier'], trained['gallery'],
                                                  payload['X'][ix], 0.363)
        result.append({'alias': r['alias'], 'sha256': r['sha256'],
                       'expected': expected, 'predicted_identity': svm_label,
                       'similarity': float(score), 'status': 'scored',
                       'faces_detected': r['faces_detected'], 'timestamp_utc': now()})
    for r in payload['failures']:
        if r['split'] == split:
            result.append({'alias': r['alias'], 'sha256': r['sha256'],
                           'expected': r['alias'] if r['alias'].startswith('K') else 'Unknown',
                           'predicted_identity': None, 'similarity': None,
                           'status': ('feature_failure' if r['reason']=='feature_failure' else
                                      'input_failure' if r['reason'].startswith('input_failure') else 'detection_failure'),
                           'faces_detected': r.get('faces_detected', 0),
                           'timestamp_utc': now()})
    return result


def apply_threshold(rows: list[dict], threshold: float) -> list[dict]:
    out=[]
    for r in rows:
        if r['status'] == 'scored':
            observed = r['predicted_identity'] if r['similarity'] >= threshold else 'Unknown'
        else:
            observed = ('No Face' if r['status']=='detection_failure' else
                        'Feature Error' if r['status']=='feature_failure' else 'Input Error')
        out.append({**r, 'threshold': threshold, 'observed': observed,
                    'passed': observed == r['expected'],
                    'error_type': ('none' if observed == r['expected'] else
                                   r['status'] if r['status'] != 'scored' else
                                   'false_accept' if r['expected'] == 'Unknown' else
                                   'false_reject' if observed == 'Unknown' else 'wrong_identity')})
    return out


def summarize(rows: list[dict]) -> dict:
    from collections import Counter
    c = Counter(r['error_type'] for r in rows)
    known = sum(r['expected'] != 'Unknown' for r in rows)
    unknown = len(rows) - known
    detected = sum(r['status'] == 'scored' for r in rows)
    genuine_accepted = sum(r['expected'] != 'Unknown' and r['observed'] == r['expected'] for r in rows)
    unknown_rejected = sum(r['expected'] == 'Unknown' and r['observed'] == 'Unknown' for r in rows)
    return {'total_images': len(rows), 'faces_scored': detected,
            'detection_failures': c['detection_failure'],
            'feature_failures': c['feature_failure'], 'input_failures': c['input_failure'],
            'known_test_images': known, 'unknown_test_images': unknown,
            'genuine_accepted': genuine_accepted, 'false_rejects': c['false_reject'],
            'false_accepts': c['false_accept'], 'wrong_identities': c['wrong_identity'],
            'unknown_rejected': unknown_rejected,
            'correct_labels': sum(r['passed'] for r in rows),
            'accuracy_on_images': (sum(r['passed'] for r in rows) / len(rows)) if rows else None}


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_image(item: dict, manifest: dict) -> tuple[np.ndarray, dict]:
    path=Path(manifest['image_root']) / item['source_file']
    if sha256(path) != item['sha256']:
        raise ValueError(f'Original image was changed or lost: {path}')
    img=cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f'Unreadable image: {path}')
    return img, item


def test_case(pipe: FacePipeline, image, expected, case_id: str,
              source: dict, trained: dict, threshold: float, transform: str) -> dict:
    try:
        faces=pipe.extract_all(image)
        if not faces:
            detected=getattr(pipe,'last_detector_count',0)
            observed='Feature Error' if detected else 'No Face'
            score=None
            reason=('YuNet detected a face but SFace alignment/feature extraction failed'
                    if detected else 'YuNet detected no face')
        else:
            face=max(faces, key=lambda f:f.box[2]*f.box[3])
            observed, score, _ = predict_open_set(trained['classifier'], trained['gallery'],
                                                 face.embedding, threshold)
            reason='' if observed == expected else ('Cosine threshold or SVM identity decision differed from expectation')
        return {'id': case_id, 'expected': expected,
                'faces_detected': getattr(pipe,'last_detector_count',len(faces)),
                'similarity': score, 'threshold': threshold, 'observed': observed,
                'status': 'Pass' if observed == expected else 'Fail',
                'reason': reason, 'image_sha256': source['sha256'],
                'source_alias': source['alias'], 'provenance': transform,
                'processed_bgr_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'timestamp_utc': now()}
    except (cv2.error, ValueError) as e:
        return {'id': case_id, 'expected': expected, 'faces_detected': 0, 'similarity': None,
                'threshold': threshold, 'observed': 'Error', 'status': 'Fail',
                'reason': f'{type(e).__name__}: {e}', 'image_sha256': source['sha256'],
                'source_alias': source['alias'], 'provenance': transform,
                'processed_bgr_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'timestamp_utc': now()}


def eight_cases(payload:dict, trained:dict, model_dir:Path, threshold:float,
                video:Path|None=None, video_expected:str|None=None) -> list[dict]:
    pipe=FacePipeline(model_dir)
    rows=payload['rows']
    train_known=next((r for r in rows if r['split']=='train' and r['alias'] in trained['classes']),None)
    if train_known is None:
        raise ValueError('No enrolled training example for T01')
    # Choose cases from the FULL source manifest, not successful extraction rows.
    # A failed detector must still be counted as a failed test.
    known_test=[{'alias':r['alias'],'split':r['split'],'source_file':r['file'],'sha256':r['sha256']}
                for r in payload['manifest']['images'] if r['split']=='test' and r['alias']==train_known['alias']]
    # If no independent photo survived detection, deliberately mark the corresponding cases Not Run.
    unknown_aliases=sorted({r['alias'] for r in payload['manifest']['images'] if r['split']=='test' and r['alias'].startswith('UT')})
    cases=[]
    def not_run(cid, expected, reason):
        return {'id': cid, 'expected': expected, 'faces_detected': None,
                'similarity': None, 'threshold': threshold, 'observed': 'Not Run',
                'status': 'Not Run', 'reason': reason, 'image_sha256': None,
                'source_alias': None, 'provenance': 'not available', 'timestamp_utc': now()}
    img,_=read_image(train_known,payload['manifest'])
    cases.append(test_case(pipe,img,train_known['alias'],'T01',train_known,trained,threshold,
                           'exact enrollment image, self-match; excluded from generalization accuracy'))
    if known_test:
        testimg,_=read_image(known_test[0], payload['manifest'])
        cases.append(test_case(pipe,testimg,train_known['alias'],'T02',known_test[0],trained,threshold,
                               'independent original photograph; different source file from enrollment'))
        dim=cv2.convertScaleAbs(testimg,alpha=.60,beta=0)
        cases.append(test_case(pipe,dim,train_known['alias'],'T03',known_test[0],trained,threshold,
                               'T02 original pixels x 0.60 digitally; NOT real dim room'))
        if len(known_test)>1:
            alt,_=read_image(known_test[1],payload['manifest'])
            cases.append(test_case(pipe,alt,train_known['alias'],'T04',known_test[1],trained,threshold,
                                   'another independent LFW photograph; head-pose not verified'))
        else:
            cases.append(not_run('T04',train_known['alias'],'Second independently detected test photo missing'))
        covered=testimg.copy()
        # Place eye rectangle using face landmarks if available; otherwise use approximate face-box placement.
        detections=pipe.extract_all(testimg)
        if detections:
            face=max(detections,key=lambda f:f.box[2]*f.box[3])
            eye_pts=face.landmarks[:2]
            center=eye_pts.mean(axis=0).astype(int)
            w=max(14,round(face.box[2]*.85))
            h=max(10,round(face.box[3]*.21))
            x1=max(0,center[0]-w//2); y1=max(0,center[1]-h//2)
            x2=min(covered.shape[1],x1+w); y2=min(covered.shape[0],y1+h)
        else:
            H,W=covered.shape[:2]; x1=int(W*.25);x2=int(W*.75);y1=int(H*.35);y2=int(H*.50)
        cv2.rectangle(covered,(x1,y1),(x2,y2),(0,0,0),-1)
        cases.append(test_case(pipe,covered,train_known['alias'],'T05',known_test[0],trained,threshold,
                               'synthetic black rectangle over detected eye landmarks; NOT physical occlusion'))
    else:
        for case in ('T02','T03','T04','T05'):
            cases.append(not_run(case,train_known['alias'],'No independently detected test photograph for enrolled class'))
    for case, alias in zip(('T06','T07'),unknown_aliases[:2]):
        example=next(({'alias':r['alias'],'split':r['split'],'source_file':r['file'],'sha256':r['sha256']}
                      for r in payload['manifest']['images'] if r['split']=='test' and r['alias']==alias),None)
        if example is None:
            cases.append(not_run(case,'Unknown',f'No detected image for distinct unknown identity {alias}'))
        else:
            sample,_=read_image(example,payload['manifest'])
            cases.append(test_case(pipe,sample,'Unknown',case,example,trained,threshold,
                                   f'independent original unknown identity {alias} (different from other unknown case)'))
    for case in ('T06','T07'):
        if not any(r['id']==case for r in cases):
            cases.append(not_run(case,'Unknown','Fewer than two independent unknown test identities detected'))
    if video is None:
        cases.append(not_run('T08',video_expected or 'unspecified','No prerecorded video was supplied. Never call a still photo video.'))
    elif not video_expected:
        cases.append(not_run('T08','unspecified','--video-expected must be provided for an interpretable test'))
    else:
        cap=cv2.VideoCapture(str(video)); frame=None
        if cap.isOpened():
            # Look for the first frame containing a detectable face, capped to 60 sampled frames.
            for _ in range(60):
                ok, candidate=cap.read()
                if not ok:
                    break
                if pipe.extract_all(candidate):
                    frame=candidate
                    break
        cap.release()
        if frame is None:
            cases.append(not_run('T08',video_expected,'Video unavailable or no face found in first 60 frames'))
        else:
            cases.append(test_case(pipe,frame,video_expected,'T08',
                                   {'sha256':sha256(video),'alias':'VIDEO'},trained,threshold,
                                   'actual decoded frame from locally supplied prerecorded video'))
    return sorted(cases,key=lambda r:r['id'])


def plot_outputs(output:Path, rows:list[dict], comparisons:list[dict]):
    output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([r['threshold'] for r in comparisons], [r['false_accept'] for r in comparisons],marker='o',label='False accept')
    ax.plot([r['threshold'] for r in comparisons], [r['false_reject'] for r in comparisons],marker='o',label='False reject')
    ax.plot([r['threshold'] for r in comparisons], [r['wrong_identity'] for r in comparisons],marker='o',label='Wrong identity')
    ax.set(xlabel='Cosine similarity threshold',ylabel='Number of scored validation images',title='Validation errors only (not test tuned)')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(output/'validation_thresholds.png',dpi=140);plt.close(fig)
    expected=[r['expected'] for r in rows]
    observed=[r['observed'] for r in rows]
    classes=sorted(set(expected)|set(observed))
    if classes:
        cm=confusion_matrix(expected,observed,labels=classes)
        fig,ax=plt.subplots(figsize=(max(7,len(classes)),max(6,len(classes))))
        ConfusionMatrixDisplay(cm,display_labels=classes).plot(ax=ax,xticks_rotation=45,colorbar=False)
        fig.tight_layout();fig.savefig(output/'test_confusion_matrix.png',dpi=140);plt.close(fig)


def evaluate(embeddings:Path, classifier:Path, models:Path, output:Path,
             video:Path|None=None, video_expected:str|None=None) -> dict:
    payload=joblib.load(embeddings)
    trained=joblib.load(classifier)
    validation=calculate_rows(payload,trained,'validation')
    threshold,comparisons=select_threshold(validation)
    tests=apply_threshold(calculate_rows(payload,trained,'test'),threshold)
    cases=eight_cases(payload,trained,models,threshold,video,video_expected)
    output.mkdir(parents=True,exist_ok=True)
    save_csv(output/'validation_rows.csv',apply_threshold(validation,threshold))
    save_csv(output/'test_rows.csv',tests)
    save_csv(output/'threshold_comparison.csv',comparisons)
    save_csv(output/'learning_log_T01_T08.csv',cases)
    plot_outputs(output,tests,comparisons)
    metrics={'created_utc':now(), 'selected_threshold':threshold,
             'threshold_selection_rule':'minimize validation scored errors; tie: false accepts, false rejects, lower threshold',
             'validation':summarize(apply_threshold(validation,threshold)),
             'final_test':summarize(tests), 'validation_thresholds':comparisons,
             'eight_case_status':{r['id']:r['status'] for r in cases},
             'train_counts':trained['train_counts'], 'versions':payload['versions'],
             'model_sha256':payload['model_sha256'], 'split_seed':payload['manifest']['seed'],
             'selection':{'known_identities':sum(v['group']=='known' for v in payload['manifest']['people'].values()),
                          'unknown_validation_identities':sum(v['group']=='unknown_validation' for v in payload['manifest']['people'].values()),
                          'unknown_test_identities':sum(v['group']=='unknown_test' for v in payload['manifest']['people'].values())},
             'warnings':['LFW faces are public research images, not consent records.',
                         'Metrics are exploratory on this selected dataset only.',
                         'No webcam result is asserted unless camera/video was actually used.',
                         'T01 self-match is not included in held-out accuracy.']}
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'chosen_threshold':threshold,'validation':metrics['validation'],
                      'test':metrics['final_test'],'eight_cases':metrics['eight_case_status']},indent=2))
    return metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--embeddings',type=Path,default=Path('local/lfw_embeddings.joblib'))
    p.add_argument('--classifier',type=Path,default=Path('local/lfw_svm.joblib'))
    p.add_argument('--models',type=Path,default=Path('models'))
    p.add_argument('--output',type=Path,default=Path('reports'))
    p.add_argument('--video',type=Path)
    p.add_argument('--video-expected',help='Expected label K01/... or Unknown for supplied video')
    args=p.parse_args()
    evaluate(args.embeddings,args.classifier,args.models,args.output,args.video,args.video_expected)

if __name__=='__main__':
    main()
