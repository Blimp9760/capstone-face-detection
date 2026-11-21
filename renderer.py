"""Only draws the results from brain.py. It doesn't recognize faces."""

import cv2


def draw(image, faces):
    picture = image.copy()
    for face in faces:
        x, y, w, h = face["box"]
        name = face["name"]
        confidence = face["confidence"]
        similarity = face["similarity"]

        text = f"{name} | face {confidence:.2f}"
        if similarity is not None:
            text += f" | match {similarity:.2f}"

        cv2.rectangle(picture, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(picture, text, (max(0, x), max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
    return picture
