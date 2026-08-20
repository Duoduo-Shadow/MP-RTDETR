from pathlib import Path

import cv2
import numpy as np


def draw_dets(img_bgr, dets, class_names=None):
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2, s, c = d.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        c = int(c)
        color = (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = str(c) if class_names is None else class_names[c]
        cv2.putText(out, f'{name}:{s:.2f}', (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def save_vis(path, img_bgr):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img_bgr)
