"""Convert region labels to compact boxes and draw BGR overlays."""
import cv2
import numpy as np
from .config import IMAGE_SIZE as SIZE

def fast_boxes_from_labels(label_map, pad=1, min_side=0):
    """Extract bounding boxes efficiently from a labeled segmentation map."""
    ys, xs = np.nonzero(label_map)
    if len(xs) == 0:
        return []

    ids = label_map[ys, xs]
    order = np.argsort(ids, kind='stable')
    ids, xs, ys = ids[order], xs[order], ys[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    counts = np.diff(np.r_[starts, len(ids)])
    min_x = np.minimum.reduceat(xs, starts)
    max_x = np.maximum.reduceat(xs, starts) + 1
    min_y = np.minimum.reduceat(ys, starts)
    max_y = np.maximum.reduceat(ys, starts) + 1

    boxes = []
    for n in range(len(starts)):
        if counts[n] < 8:
            continue
        side = max(max_x[n] - min_x[n], max_y[n] - min_y[n], min_side) + 2 * pad
        cx = (min_x[n] + max_x[n]) / 2
        cy = (min_y[n] + max_y[n]) / 2
        box = [int(max(0, round(cx - side / 2))), int(max(0, round(cy - side / 2))),
               int(min(SIZE, round(cx + side / 2))), int(min(SIZE, round(cy + side / 2)))]
        if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
            boxes.append(box)
    return boxes


def draw_boxes(image_bgr,boxes,color,labels=None,thickness=2):
    """Draw rectangles on a BGR copy."""
    view=image_bgr.copy()
    for i,(x1,y1,x2,y2) in enumerate(boxes,1):
        cv2.rectangle(view,(x1,y1),(x2,y2),color,thickness)
        if labels is not None:
            cv2.putText(view,str(labels[i-1]),(x1,max(y1-5,12)),
                        cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    return view
