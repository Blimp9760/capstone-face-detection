# Privacy, ethics, and retention

This is an educational exploration, **not** a system for real-world identity verification, public camera monitoring, attendance, security, misconduct decisions, or any consequential identification.

## LFW terms

Labeled Faces in the Wild (LFW) contains photographs originally collected from publicly available imagery. It is an academic face-recognition research benchmark. The collection does not demonstrate permission from every depicted person for biometric enrollment. **LFW does not have a clear blanket grant allowing unrestricted reuse of every photo**. The program downloads research images to local disk and avoids republishing them. Review dataset-owner terms and school ethical requirements before any external use. Citation: Gary B. Huang et al., *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*, 2007 (https://vis-www.cs.umass.edu/lfw/).

OpenCV Zoo source is Apache-2.0 at repository level; model subfolders may also have model-specific notices. Check model-folder `LICENSE` files for use/distribution restrictions (https://github.com/opencv/opencv_zoo).

## Personal data and deletion

- Use the live camera ONLY with participant consent; do not take or save other people's face images without permission.
- The webcam UI stores normalized face embeddings under `local/enrollments.json`; these are still biometric data even without saved photographs.
- Source webcam frames are kept in memory for processing and not sent to a web service by this project.
- Deletion removes the identity and vectors from the current JSON file; it does **not** guarantee secure erasure from OS temporary files, SSD wear leveling, earlier backup copies, memory, or screenshots.
- No default retention schedule is enforced. Define and honor one before allowing real-person enrollment; when finished, delete local files and unrelated external backup copies according to applicable obligations.
- `data/` contains LFW JPEGs and identity mapping; `local/` contains private biometric vectors. Neither belongs in a public or private GitHub repository.

## Scientific caution

Recognition is fallible and may work differently across demographic groups, camera quality and illumination. The selected dataset is small and nonrepresentative. Cosine similarity measures feature proximity, not probabilistic confidence or actual identity certainty. The closed-set SVM always chooses a known class; only the separate cosine threshold produces Unknown.

Do not cite peer interviews, teacher quotations or user-participant feedback without a real record and permission. Do not reinterpret the Learning Log's sample values as verified measurements.
