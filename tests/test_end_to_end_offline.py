"""Exercise complete dataset -> trained SVM -> real CSV/plots without downloading biometric images.

All frames below are synthetic solid-color arrays and the OpenCV face network is
replaced with a tiny deterministic fake. This checks code plumbing, NOT accuracy.
"""
import math
import cv2
import numpy as np
from facecap.dataset import make_manifest
from facecap.pipeline import Face
from facecap.train import extract_manifest, train_svm
from facecap.evaluate import evaluate


class FakeFacePipeline:
    def __init__(self, *args, **kwargs):
        pass

    def extract_all(self, image):
        v=float(image[0,0,0])
        a=v/255 * math.pi
        feature=np.array([math.cos(a),math.sin(a)],dtype=np.float32)
        face=Face(box=(0,0,20,20),confidence=1.0,embedding=feature,
                  landmarks=np.array([[4,8],[15,8],[10,12],[5,16],[15,16]],dtype=np.float32))
        return [face]

    def extract_one(self, image):
        f=self.extract_all(image)[0]
        return f.embedding,1,f


def test_all_outputs_with_fake_faces_only(tmp_path,monkeypatch):
    import facecap.train as train_module
    import facecap.evaluate as eval_module
    monkeypatch.setattr(train_module,'FacePipeline',FakeFacePipeline)
    monkeypatch.setattr(eval_module,'FacePipeline',FakeFacePipeline)
    dataset=tmp_path/'images'
    for person in range(5):
        folder=dataset/f'person_{person:02d}'
        folder.mkdir(parents=True)
        for j in range(8):
            img=np.full((24,24,3),40+person*40+j%2,dtype=np.uint8)
            assert cv2.imwrite(str(folder/f'capture_{j:02d}.jpg'),img)
    manifest=tmp_path/'data'/'manifest.json'
    make_manifest(dataset,manifest,known_count=2,unknown_validation_count=1,
                  unknown_test_count=2,min_images=8)
    embs=tmp_path/'local'/'vectors.joblib'
    svm=tmp_path/'local'/'model.joblib'
    payload=extract_manifest(manifest,tmp_path/'models',embs)
    trained=train_svm(payload,svm)
    assert len(trained['classes'])==2
    metrics=evaluate(embs,svm,tmp_path/'models',tmp_path/'reports')
    assert metrics['final_test']['total_images']>0
    assert set(metrics['eight_case_status'])=={f'T{i:02d}' for i in range(1,9)}
    assert metrics['eight_case_status']['T08']=='Not Run'
    for filename in ['metrics.json','test_rows.csv','validation_rows.csv',
                     'threshold_comparison.csv','learning_log_T01_T08.csv',
                     'test_confusion_matrix.png','validation_thresholds.png']:
        assert (tmp_path/'reports'/filename).is_file(),filename
