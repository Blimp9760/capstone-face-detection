import numpy as np
import pytest
from sklearn.svm import SVC
from facecap.recognition import select_threshold, predict_open_set, THRESHOLDS
from facecap.evaluate import summarize, apply_threshold


def test_threshold_only_validation_and_raw_scores():
    rows=[{'status':'scored','expected':'K01','predicted_identity':'K01','similarity':.8},
          {'status':'scored','expected':'Unknown','predicted_identity':'K01','similarity':.4}]
    chosen,table=select_threshold(rows)
    assert chosen == pytest.approx(.7)
    assert [t['threshold'] for t in table]==list(THRESHOLDS)
    assert table[1]['correct']==2


def test_detect_failure_not_conflated_with_rejection():
    rows=[{'status':'detection_failure','expected':'K01','predicted_identity':None,'similarity':None},
          {'status':'scored','expected':'Unknown','predicted_identity':'K01','similarity':.1}]
    scored=apply_threshold(rows,.7)
    assert scored[0]['observed']=='No Face'
    assert scored[0]['error_type']=='detection_failure'
    assert scored[1]['observed']=='Unknown'
    assert summarize(scored)['detection_failures']==1


def test_predict_svm_then_gate_by_gallery():
    X=np.array([[1.,0],[.95,.05],[0,1],[.05,.95]],dtype=np.float32)
    y=['K01','K01','K02','K02']
    clf=SVC(kernel='linear').fit(X,y)
    gallery={'K01':X[:2], 'K02':X[2:]}
    assert predict_open_set(clf,gallery,np.array([1.,0]),.9)[0]=='K01'
    assert predict_open_set(clf,gallery,np.array([-.2,-1.]),.9)[0]=='Unknown'
