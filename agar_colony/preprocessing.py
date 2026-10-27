"""Text suppression and local agar-background estimation."""
import cv2
import numpy as np
from .config import IMAGE_SIZE as SIZE

def prepare_image(base, mask):
    """Suppress printed text and set pixels outside the valid ROI to a flat color."""
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
    text = cv2.inRange(hsv, np.array([50,35,35],np.uint8), np.array([92,85,165],np.uint8))
    text = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    text = cv2.bitwise_and(text, mask)
    if cv2.countNonZero(text):
        base = cv2.inpaint(base, text, 3, cv2.INPAINT_TELEA)
    inside = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    pixels = base[inside > 0]
    median_bgr = np.median(pixels[::5],axis=0).astype(np.uint8) if pixels.size else np.array([128]*3,np.uint8)
    result = np.where(inside[:,:,None] > 0, base, median_bgr[None,None,:])
    rim = cv2.subtract(mask,inside)
    return np.where(rim[:,:,None] > 0, cv2.GaussianBlur(result,(5,5),0),result)


def estimate_background(lab, mask, kernel=121):
    """Estimate a slowly varying agar background with median and Gaussian filters."""
    valid = mask > 0
    filled = lab.copy()
    if valid.any():
        filled[~valid] = np.median(lab[valid],axis=0).astype(np.uint8)
    small = cv2.resize(filled,(SIZE//4,SIZE//4),interpolation=cv2.INTER_AREA)
    size = max(3, (int(kernel)//4)|1)
    background = cv2.medianBlur(small,size)
    background = cv2.GaussianBlur(background,(0,0),max(1.0,size/6))
    return cv2.resize(background,(SIZE,SIZE),interpolation=cv2.INTER_LINEAR)
