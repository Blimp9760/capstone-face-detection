"""Draw face boxes and names. All recognition happens in brain.py."""

import cv2


def draw(image, faces):
    image = image.copy()
    if not faces:
        cv2.putText(image, "No face detected", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
    for face in faces:
        x, y, w, h = face["box"]
        label = f'{face["name"]} | face {face["confidence"]:.2f}'
        if face["similarity"] is not None:
            label += f' | match {face["similarity"]:.2f}'
        color = (0, 0, 255) if face["name"] == "Unknown" else (0, 200, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (max(0, x), max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image
