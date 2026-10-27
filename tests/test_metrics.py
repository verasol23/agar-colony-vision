"""Pure-Python tests for bounding box conventions and matching."""
from agar_colony.evaluation import bbox_iou, match_one_to_one, summarize
from agar_colony.annotations import work_to_original, write_yolo_annotations
from agar_colony.batch import summary_from_rows


def test_iou_and_duplicate_matching():
    assert bbox_iou([0,0,10,10],[0,0,10,10])==1.0
    assert bbox_iou([0,0,10,10],[20,20,30,30])==0.0
    stats,(pairs,fp,fn) = summarize([[0,0,10,10],[0,0,10,10]],[[0,0,10,10]],0.5)
    assert (stats['TP'],stats['FP'],stats['FN'])==(1,1,0)
    assert len(pairs)==1


def test_no_json_rows_excluded_from_summary():
    rows = [dict(has_annotation=True,TP=3,FP=1,FN=1),
            dict(has_annotation=False,TP=None,FP=None,FN=None)]
    summary=summary_from_rows(rows)
    assert len(summary) == 1
    assert summary[0]['F1'] == 0.75
    assert summary[0]['Evaluated_images'] == 1


def test_coordinate_roundtrip_and_yolo(tmp_path):
    geom={'x0':100, 'y0':200, 'scale':0.5}
    assert work_to_original([10,20,30,40],geom)==[120,240,160,280]
    dst=tmp_path/'image.txt'
    assert write_yolo_annotations(dst,[{'class':'B.subtilis','box':[0,0,512,512]}])==1
    assert dst.read_text().startswith('1 0.250000 0.250000 0.500000 0.500000')
