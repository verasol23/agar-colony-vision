"""Rendering overlays and intermediate visualizations."""
import cv2
import numpy as np
from .boxes import draw_boxes
from .evaluation import evaluation_overlay


def render_detection(crop_bgr, boxes):
    """Draw numbered, unclassified predictions on a copy of the cropped dish."""
    return draw_boxes(crop_bgr, boxes, (0, 0, 255), list(range(1, len(boxes)+1)))


def render_evaluation(crop_bgr, boxes, gt, match_data):
    """GT blue, matched predictions green, false positives red, false negatives orange."""
    return evaluation_overlay(crop_bgr, boxes, gt, match_data)


def render_stage(stage_key, stages):
    """Convert integer watershed labels or raw masks into BGR for Tk preview."""
    arr = stages[stage_key]
    if arr.dtype == np.int32 or arr.dtype == np.int64:
        hue = ((arr.astype(np.int64) * 47) % 180).astype(np.uint8)
        val = np.where(arr > 0, 255, 0).astype(np.uint8)
        hsv = np.dstack([hue, np.full_like(hue, 190), val])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return arr.copy()
