import numpy as np
from facecap.store import EnrollmentStore


def test_enrollment_survives_restart_then_delete(tmp_path):
    loc=tmp_path/'store.json'
    a=EnrollmentStore(loc)
    assert a.recognize([1,0],.363)[0]=='Unknown'
    a.enroll('Example consenting user',[1,0])
    a.enroll('Example consenting user',[.95,.05])
    assert a.list_names()==['Example consenting user']
    b=EnrollmentStore(loc)
    assert b.recognize([1,0],.7)[0]=='Example consenting user'
    assert b.delete('Example consenting user') is True
    c=EnrollmentStore(loc)
    assert c.list_names()==[]
    assert c.recognize([1,0],.7)[0]=='Unknown'
    assert c.delete('Example consenting user') is False


def test_unknown_rejection(tmp_path):
    s=EnrollmentStore(tmp_path/'store.json')
    s.enroll('A',[1,0])
    assert s.recognize([0,1],.7)[0]=='Unknown'


def test_cannot_enroll_blank_name(tmp_path):
    s=EnrollmentStore(tmp_path/'store.json')
    import pytest
    with pytest.raises(ValueError):s.enroll('   ',[1,0])
