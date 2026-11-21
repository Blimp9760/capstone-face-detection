"""Closed-set SVM prediction gated by open-set cosine similarity."""
import numpy as np
from .pipeline import similarity, unit

THRESHOLDS = (0.363, 0.700, 0.850, 0.900, 0.950)


def predict_open_set(classifier, gallery: dict[str, list[np.ndarray]], embedding, threshold: float):
    """Choose SVM identity, then require similarity to an example from that class."""
    prediction = str(classifier.predict(unit(embedding).reshape(1, -1))[0])
    candidates = gallery.get(prediction, [])
    if len(candidates) == 0:
        return "Unknown", float("nan"), prediction
    best = max(similarity(embedding, v) for v in candidates)
    return prediction if best >= threshold else "Unknown", best, prediction


def select_threshold(rows: list[dict], thresholds=THRESHOLDS) -> tuple[float, list[dict]]:
    """Tune with validation ONLY. Prioritize fewer false accepts at equal total error."""
    table = []
    for t in thresholds:
        false_accept = false_reject = misidentified = evaluated = correct = 0
        for row in rows:
            if row.get("status") != "scored":
                continue
            evaluated += 1
            expected = row["expected"]
            prediction = row["predicted_identity"]
            label = prediction if row["similarity"] >= t else "Unknown"
            correct += (label == expected)
            if expected == "Unknown" and label != "Unknown":
                false_accept += 1
            elif expected != "Unknown" and label == "Unknown":
                false_reject += 1
            elif expected != "Unknown" and label != expected:
                misidentified += 1
        table.append({"threshold": float(t), "evaluated": evaluated, "correct": correct,
                      "false_accept": false_accept, "false_reject": false_reject,
                      "wrong_identity": misidentified})
    winner = min(table, key=lambda r: (r["false_accept"] + r["false_reject"] + r["wrong_identity"],
                                       r["false_accept"], r["false_reject"], r["threshold"]))
    return winner["threshold"], table
