"""Single-image detection, independent of annotation data."""
import cv2
import numpy as np
from .config import DetectorConfig, IMAGE_SIZE as SIZE
from .dish import detect_dish, crop_dish, find_agar_boundary, ellipse_mask
from .preprocessing import prepare_image, estimate_background
from .segmentation import hysteresis_mask, exclude_rim_artifacts, clean_mask, split_colonies
from .edge import detect_edge_colonies
from .boxes import fast_boxes_from_labels
from .evaluation import bbox_iou

def detect_colonies(image_bgr, config=None, include_stages=False):
    """Run colony detection independently from ground-truth annotations."""
    config = config or DetectorConfig()
    cx, cy, radius, dish_method = detect_dish(image_bgr)
    crop_bgr, geom = crop_dish(image_bgr, cx, cy, radius, SIZE)
    shape = find_agar_boundary(crop_bgr)
    roi = ellipse_mask(shape, config.dish_shrink / 100.0)
    base = cv2.bitwise_and(crop_bgr, crop_bgr, mask=roi)

    prepared = prepare_image(base, roi)
    blur = (cv2.GaussianBlur(prepared, (0, 0), config.gaussian_sigma)
            if config.gaussian_sigma >= 0.3 else prepared.copy())
    lab_image = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    background = estimate_background(lab_image, roi, config.bg_kernel)

    # Shrink the valid analysis region away from interpolation artifacts.
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    valid = cv2.erode(cv2.erode(roi, small_kernel), small_kernel)
    diff = lab_image.astype(np.float32) - background.astype(np.float32)
    diff *= (valid > 0)[:, :, None]
    distance = np.sqrt((config.l_weight * diff[:, :, 0]) ** 2
                       + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)

    values = distance[valid > 0]
    if values.size == 0:
        raise ValueError('Dish ROI contains no valid pixels')
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    noise = max(1.4826 * mad, 1.0)
    hyst = hysteresis_mask(distance, valid, med + config.thr_low * noise,
                           med + config.thr_high * noise, config.multiscale_sigma)

    # Suppress the HSV range associated with printed text.
    hsv = cv2.cvtColor(prepared, cv2.COLOR_BGR2HSV)
    ink = cv2.inRange(hsv, np.array([50, 35, 35], np.uint8),
                     np.array([92, 85, 165], np.uint8))
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    hyst[ink > 0] = 0

    without_rim = exclude_rim_artifacts(hyst, roi)
    cleaned = clean_mask(without_rim, opening_radius=config.opening_radius,
                         max_hole_area=config.hole_max, min_area=config.min_area,
                         max_area=config.max_area)
    labels, _dist = split_colonies(cleaned, blur, config.ws_smooth, config.ws_alpha)

    # The inner-dish candidates are a required stage of the single pipeline.
    inner_boxes = fast_boxes_from_labels(labels, pad=config.box_pad)
    edge_mask = np.zeros((SIZE, SIZE), np.uint8)
    edge_boxes = []
    if shape is not None:
        edge_mask, edge_boxes = detect_edge_colonies(crop_bgr, shape, config)

    # Preserve original box sizes; only append genuinely new edge candidates.
    boxes = inner_boxes.copy()
    edge_added = 0
    for candidate in edge_boxes:
        if all(bbox_iou(candidate, existing) < config.dedup_iou for existing in boxes):
            boxes.append(candidate)
            edge_added += 1

    result = {
        'crop': crop_bgr, 'geometry': geom, 'dish_method': dish_method,
        'boxes': boxes, 'inner_count': len(inner_boxes), 'edge_added': edge_added,
        'mask': roi, 'hysteresis': hyst, 'morphology': cleaned,
        'edge_mask': edge_mask, 'edge_candidates': len(edge_boxes),
    }
    if include_stages:
        result['stages'] = {
            'Cropped dish': crop_bgr, 'Dish mask': roi, 'Prepared image': prepared,
            'Estimated background': cv2.cvtColor(background, cv2.COLOR_LAB2BGR),
            'Color distance': cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            'Hysteresis': hyst, 'Morphology': cleaned,
            'Watershed labels': labels, 'Edge mask': edge_mask,
        }
    return result
