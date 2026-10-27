"""Exploratory HSV annular detector for colonies close to dish boundaries."""
import cv2
import numpy as np
from .config import IMAGE_SIZE as SIZE, DetectorConfig

def elliptical_radius_field(shape):
    """Return relative pixel distance to the fitted ellipse boundary."""
    (cx,cy),(width,height),angle=shape
    yy,xx=np.ogrid[:SIZE,:SIZE]
    dx,dy=xx-cx,yy-cy
    theta=np.deg2rad(angle)
    ux=dx*np.cos(theta)+dy*np.sin(theta)
    uy=-dx*np.sin(theta)+dy*np.cos(theta)
    return np.sqrt((ux/(width/2))**2+(uy/(height/2))**2)


def detect_edge_colonies(crop_bgr,shape,config=None):
    """Search the agar rim for compact high-saturation brown colonies."""
    config = config or DetectorConfig()
    hsv=cv2.cvtColor(cv2.GaussianBlur(crop_bgr,(5,5),0),cv2.COLOR_BGR2HSV)
    hue,sat,val=cv2.split(hsv)
    relative_r=elliptical_radius_field(shape)
    # The annulus is wider than the original inner-dish mask.
    selected=((relative_r>=config.edge_r_min)&(relative_r<=config.edge_r_max)&
              (hue>=config.edge_h_low)&(hue<=config.edge_h_high)&
              (sat>=config.edge_s_min)&(val>=config.edge_v_low)&(val<=config.edge_v_high))
    edge_mask=selected.astype(np.uint8)*255
    edge_mask=cv2.morphologyEx(edge_mask,cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    edge_mask=cv2.morphologyEx(edge_mask,cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    count,labels,stats,_=cv2.connectedComponentsWithStats(edge_mask,connectivity=8)
    boxes=[]
    for i in range(1,count):
        x,y,w,h,area=map(int,stats[i])
        aspect=min(w,h)/max(w,h)
        extent=area/(w*h)
        if config.edge_min_area<=area<=config.edge_max_area and 18<=w<=95 and 18<=h<=95 and aspect>=.55 and extent>=.45:
            boxes.append([x,y,x+w,y+h])
    return edge_mask,boxes
