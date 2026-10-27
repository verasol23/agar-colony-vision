"""Color-change hysteresis, morphology and watershed separation."""
import cv2
import numpy as np
from .config import IMAGE_SIZE as SIZE

def hysteresis_mask(distance, valid, low, high, smooth_sigma=2.0):
    """Keep weak components only when connected to a strong pixel."""
    inside = valid>0
    def propagate(data, low_limit, high_limit):
        weak = ((data>=low_limit)&inside).astype(np.uint8)
        count, components = cv2.connectedComponents(weak,connectivity=8)
        strong_ids = np.unique(components[(data>=high_limit)&inside])
        selected = np.zeros(count,bool)
        selected[strong_ids] = True
        selected[0] = False
        return selected[components].astype(np.uint8)*255
    fine = propagate(distance,low,high)
    smooth = (cv2.GaussianBlur(distance,(0,0),smooth_sigma)
              if smooth_sigma >= 0.3 else distance)
    coarse = propagate(smooth,.85*low,.85*high)
    return cv2.bitwise_or(fine,coarse)


def exclude_rim_artifacts(binary, valid_mask):
    """Remove long or diffuse components that follow the dish boundary."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary,connectivity=8)
    if count<2:
        return binary.copy()
    edge = cv2.distanceTransform((valid_mask>0).astype(np.uint8),cv2.DIST_L2,3)
    total = stats[:,cv2.CC_STAT_AREA].astype(float)
    counts = np.bincount(labels[(edge<.035*SIZE)&(labels>0)].ravel(),minlength=count)
    near_fraction = counts / np.maximum(total,1)
    w = stats[:,cv2.CC_STAT_WIDTH].astype(float)
    h = stats[:,cv2.CC_STAT_HEIGHT].astype(float)
    aspect = np.maximum(w,h)/np.maximum(np.minimum(w,h),1)
    fill = total / np.maximum(w*h,1)
    bad = (near_fraction>.15)&((aspect>2.5)|(fill<.35))
    interior = near_fraction[1:]<.05
    typical = float(np.median(total[1:][interior])) if np.any(interior) else 0
    if typical>0:
        bad |= (near_fraction>.15)&(total>4*typical)
    keep = ~bad
    keep[0] = False
    return keep[labels].astype(np.uint8)*255


def clean_mask(binary, opening_radius=2, max_hole_area=300,
               min_area=8, max_area=25000):
    """Apply opening, fill small holes and remove abnormal components."""
    k=opening_radius*2+1
    result=cv2.morphologyEx(binary,cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    contours,hierarchy=cv2.findContours(result,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    hole_mask=np.zeros_like(result)
    if hierarchy is not None:
        for contour,info in zip(contours,hierarchy[0]):
            if info[3]>=0 and cv2.contourArea(contour)<=max_hole_area:
                cv2.drawContours(hole_mask,[contour],-1,255,cv2.FILLED)
    filled=cv2.bitwise_or(result,hole_mask)
    count,labels,stats,_=cv2.connectedComponentsWithStats(filled,connectivity=8)
    areas=stats[:,cv2.CC_STAT_AREA]
    keep=(areas>=min_area)&(areas<=max_area)
    keep[0]=False
    return keep[labels].astype(np.uint8)*255


def split_colonies(binary,color, smooth=1.0, alpha=1.0):
    """Split touching regions using distance maxima and OpenCV watershed."""
    count,components=cv2.connectedComponents(binary,connectivity=8)
    if count<2:
        return components.astype(np.int32),np.zeros_like(binary,dtype=np.float32)
    dist=cv2.distanceTransform(binary,cv2.DIST_L2,5)
    if smooth>=.3:
        dist=cv2.GaussianBlur(dist,(0,0),smooth)
    maxima=(dist>=cv2.dilate(dist,np.ones((3,3),np.uint8)))&(dist>=1.5)&(binary>0)
    ys,xs=np.nonzero(maxima)
    radii=dist[ys,xs]
    order=np.argsort(-radii)[:4000]
    ys,xs,radii=ys[order],xs[order],radii[order]
    seeds=[]
    for x,y,r in zip(xs,ys,radii):
        if any((px-x)**2+(py-y)**2<(alpha*pr)**2 for px,py,pr in seeds):
            continue
        seeds.append((int(x),int(y),float(r)))
    markers=np.zeros(binary.shape,np.int32)
    for index,(x,y,r) in enumerate(seeds,2):
        cv2.circle(markers,(x,y),1,index,-1)
    # Supply one marker for each valid component lacking a peak.
    seeded=set(components[markers>=2].tolist())
    next_label=len(seeds)+2
    for component_id in range(1,count):
        if component_id not in seeded:
            markers[components==component_id]=next_label
            next_label+=1
    markers[binary==0]=1
    cv2.watershed(color,markers)
    return np.where((markers>=2)&(binary>0),markers-1,0).astype(np.int32),dist
