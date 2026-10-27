"""Dish detection, standardization and coordinate mapping."""
import math
import cv2
import numpy as np
from .config import IMAGE_SIZE as SIZE

def detect_dish(image_bgr):
    """Find the approximate dish circle using Otsu, then Hough as fallback."""
    h, w = image_bgr.shape[:2]
    ratio = min(1.0, 1000.0 / max(h, w))
    small = cv2.resize(image_bgr, None, fx=ratio, fy=ratio,
                       interpolation=cv2.INTER_AREA) if ratio < 1 else image_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corners = np.concatenate((binary[:10, :10].ravel(), binary[-10:, -10:].ravel()))
    if corners.mean() > 128:
        binary = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        filling = cv2.contourArea(largest) / (np.pi * radius**2 + 1e-9)
        if 0.25 * min(gray.shape) <= radius <= 0.75 * max(gray.shape) and filling >= 0.75:
            return cx / ratio, cy / ratio, radius / ratio, 'Otsu'
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=2,
                               minDist=min(gray.shape), param1=100, param2=60,
                               minRadius=int(.3 * min(gray.shape)),
                               maxRadius=int(.6 * max(gray.shape)))
    if circles is not None:
        cx, cy, radius = circles[0][0]
        return float(cx / ratio), float(cy / ratio), float(radius / ratio), 'Hough'
    return w / 2, h / 2, .48 * min(h, w), 'Fallback'


def crop_dish(image_bgr, cx, cy, radius, size=1024):
    """Crop a square around the detected dish and store coordinate mapping."""
    half = max(8, int(round(radius)))
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    side = half * 2
    canvas = np.zeros((side, side, 3), np.uint8)
    h, w = image_bgr.shape[:2]
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = image_bgr[sy0:sy1, sx0:sx1]
    crop = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return crop, {'x0': x0, 'y0': y0, 'scale': size / side}


def find_agar_boundary(crop_bgr):
    """Estimate the inner agar ring by polar analysis and ellipse fitting."""
    size = crop_bgr.shape[0]
    half = size / 2.0
    nr, na = int(half), 360
    gray = cv2.GaussianBlur(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    polar = cv2.warpPolar(gray, (nr, na), (half, half), half,
                          cv2.WARP_POLAR_LINEAR).astype(np.float32)
    contrast = np.abs(cv2.blur(polar, (15, 1)) - polar)
    profile = np.convolve(np.median(contrast, axis=0), np.ones(5) / 5, mode='same')
    lo, hi = int(.55 * nr), int(.97 * nr)
    seg = profile[lo:hi]
    peak = float(seg.max()) if seg.size else 0.0
    if peak < 2.5:
        return None
    ring = None
    for r in range(5, len(seg)-5):
        if seg[r] >= .45 * peak and seg[r] == seg[r-5:r+6].max():
            ring = lo+r
            break
    if ring is None:
        return None
    default = ((half, half), (2.0*ring, 2.0*ring), 0.0)
    window = max(6, int(.02 * nr))
    band = contrast[:, ring-window:ring+window+1]
    radii = (ring-window+band.argmax(axis=1)).astype(np.float32)
    ok = band.max(axis=1) >= .35 * profile[ring]
    extension = np.r_[radii[-10:], radii, radii[:10]]
    moving_median = np.array([np.median(extension[i:i+21]) for i in range(na)])
    ok &= np.abs(radii-moving_median) <= 4
    if ok.sum() < .4*na:
        return default
    angles = np.deg2rad(np.arange(na))
    points = np.c_[half+radii*np.cos(angles), half+radii*np.sin(angles)][ok].astype(np.float32)
    (center_x, center_y), (width, height), angle = cv2.fitEllipse(points)
    valid = (abs(center_x-half) < .12*size and abs(center_y-half) < .12*size
             and .55*size < (width+height)/2 < .99*size
             and max(width,height)/max(min(width,height),1) < 1.12)
    return ((float(center_x), float(center_y)),
            (float(width), float(height)), float(angle)) if valid else default


def ellipse_mask(shape, ratio=1.0):
    """Create an ROI using the estimated agar ellipse."""
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    if shape is None:
        shape = ((SIZE/2, SIZE/2), (.87*SIZE, .87*SIZE), 0.0)
    (ex, ey), (ew, eh), angle = shape
    cv2.ellipse(mask, ((ex, ey), (ew*ratio, eh*ratio), angle), 255, -1)
    return mask


def original_to_work(box, geom, size=SIZE):
    """Map a bounding box from original pixels into 1024-pixel crop space."""
    x1, y1, x2, y2 = box
    values = [(v - offset)*geom['scale'] for v, offset in
              zip([x1,y1,x2,y2], [geom['x0'],geom['y0'],geom['x0'],geom['y0']])]
    return [int(round(np.clip(v, 0, size))) for v in values]
