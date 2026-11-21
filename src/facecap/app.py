"""Local webcam UI: e enroll, d delete, l list, q quit. No uploads."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from .pipeline import FacePipeline
from .store import EnrollmentStore


def recognize_image(path:Path, model_dir:Path, store_path:Path, threshold:float):
    pipe=FacePipeline(model_dir)
    img=cv2.imread(str(path))
    if img is None:
        raise ValueError(f'Unable to open image: {path}')
    store=EnrollmentStore(store_path)
    faces=pipe.extract_all(img)
    results=[]
    for f in faces:
        label,score=store.recognize(f.embedding,threshold)
        results.append({'name':label,'similarity':score,'box':f.box})
    print(f'Local processing | faces={len(faces)} threshold={threshold} | {results}')
    return results


def webcam(model_dir:Path, store_path:Path, threshold:float, camera_index:int=0):
    pipe=FacePipeline(model_dir)
    store=EnrollmentStore(store_path)
    cap=cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError('Could not open local webcam. Check OS camera permissions and camera index.')
    print('Camera only processed locally; e = enroll largest face, d = delete name, l = list, q = quit')
    print('IMPORTANT: get the person\'s permission before storing their face embedding.')
    last_faces=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:
                print('Camera read failed')
                break
            faces=pipe.extract_all(frame)
            last_faces=faces
            for f in faces:
                name,score=store.recognize(f.embedding,threshold)
                x,y,w,h=f.box
                cv2.rectangle(frame,(x,y),(x+w,y+h),(20,225,100) if name!='Unknown' else (30,180,240),2)
                label=f'{name} | similarity {score:.3f}' if score==score else 'Unknown | no enrollment'
                cv2.putText(frame,label,(max(0,x),max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
            h,w=frame.shape[:2]
            cv2.rectangle(frame,(0,0),(w,66),(25,25,25),-1)
            cv2.putText(frame,f'LOCAL PROCESSING | threshold {threshold:.3f} | known: {len(store.list_names())}',
                        (10,24),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),1)
            cv2.putText(frame,'E Enroll   D Delete   L List   Q Quit',
                        (10,49),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            cv2.imshow('Capstone Local Face Recognition',frame)
            key=cv2.waitKey(1) & 0xff
            if key in (ord('q'),27):
                break
            if key==ord('l'):
                print('Locally enrolled:',store.list_names())
            if key==ord('e'):
                if not last_faces:
                    print('No face detected; enrollment cancelled')
                    continue
                face=max(last_faces,key=lambda f:f.box[2]*f.box[3])
                name=input('Enrolment name (permission required): ').strip()
                if not name:
                    print('Cancelled');continue
                permission=input(f'Confirm {name} consented to LOCAL biometric enrollment [yes/no]: ').strip().lower()
                if permission!='yes':
                    print('Cancelled; no embedding saved');continue
                store.enroll(name,face.embedding)
                print(f'Enrollment saved locally for {name}. Raw image was not saved.')
            if key==ord('d'):
                name=input('Exact identity to delete: ').strip()
                if name not in store.list_names():
                    print('Not found');continue
                if input(f'Type DELETE to remove all local embeddings for {name}: ').strip()=='DELETE':
                    store.delete(name);print('Deleted from application enrollment file (not secure disk erasure).')
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path,default=Path('models'))
    p.add_argument('--store',type=Path,default=Path('local/enrollments.json'))
    p.add_argument('--threshold',type=float,default=.363)
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--image',type=Path)
    a=p.parse_args()
    if not -1<=a.threshold<=1: raise ValueError('Cosine threshold must be in [-1,1]')
    if a.image: recognize_image(a.image,a.models,a.store,a.threshold)
    else: webcam(a.models,a.store,a.threshold,a.camera)

if __name__=='__main__':
    main()
