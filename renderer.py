"""Draw face boxes and names. All recognition happens in brain.py."""

import cv2


def draw(image, faces):
    image = image.copy()
    for face in faces:
        x, y, w, h = face["box"]
        label = f'{face["name"]} | face {face["confidence"]:.2f}'
        if face["similarity"] is not None:
            label += f' | match {face["similarity"]:.2f}'
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(image, label, (max(0, x), max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
    return image
