"""Standard IoU-based one-to-one object-detection evaluation."""
import cv2
from .boxes import draw_boxes

def bbox_iou(a,b):
    """Calculate standard intersection-over-union (without containment bonus)."""
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    union=aa+ab-intersection
    return intersection/union if union else 0.0


def match_one_to_one(pred,gt,threshold=.5):
    """Greedily match each prediction to at most one ground-truth box."""
    choices=sorted([(bbox_iou(p,g),i,j) for i,p in enumerate(pred)
                    for j,g in enumerate(gt)],reverse=True)
    matches=[];used_pred=set();used_gt=set()
    for iou,i,j in choices:
        if iou<threshold:
            break
        if i not in used_pred and j not in used_gt:
            matches.append((i,j,iou));used_pred.add(i);used_gt.add(j)
    fp=[i for i in range(len(pred)) if i not in used_pred]
    fn=[j for j in range(len(gt)) if j not in used_gt]
    return matches,fp,fn


def summarize(pred,gt,threshold=.5):
    matches,fp,fn=match_one_to_one(pred,gt,threshold)
    tp=len(matches)
    precision=tp/len(pred) if pred else 0.0
    recall=tp/len(gt) if gt else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'GT':len(gt),'Pred':len(pred),'TP':tp,'FP':len(fp),'FN':len(fn),
            'Precision':precision,'Recall':recall,'F1':f1},(matches,fp,fn)


def evaluation_overlay(img, predictions,truth,matched_data):
    """Draw TP green, FP red, FN orange and matched GT thin blue."""
    matches,fp,fn=matched_data
    output=img.copy()
    for i,j,iou in matches:
        output=draw_boxes(output,[truth[j]],(255,0,0),thickness=1)
        output=draw_boxes(output,[predictions[i]],(0,190,0),[f'TP {iou:.2f}'],2)
    for i in fp:
        output=draw_boxes(output,[predictions[i]],(0,0,255),['FP'],2)
    for j in fn:
        output=draw_boxes(output,[truth[j]],(0,165,255),['FN'],2)
    return output
