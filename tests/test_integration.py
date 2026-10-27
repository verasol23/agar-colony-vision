import pytest
"""Integration tests for the single Improved detector and manual review."""
import csv
from pathlib import Path
import shutil
import cv2
from agar_colony.config import DetectorConfig
from agar_colony.detection import detect_colonies
from agar_colony.annotations import read_agar_annotation, save_reviewed_annotations
from agar_colony.evaluation import summarize
from agar_colony.batch import run_batch

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'countable'


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "examples" / "countable" / "1000.jpg").exists(), reason="AGAR sample image is distributed separately")
def test_improved_regression_on_sample():
    image = cv2.imread(str(EXAMPLE / '1000.jpg'))
    result = detect_colonies(image, DetectorConfig(), include_stages=True)
    assert 'baseline' not in result and 'improved' not in result
    assert len(result['boxes']) == 4
    assert result['inner_count'] == 3
    assert result['edge_added'] == 1
    assert 'Watershed labels' in result['stages']
    gt = read_agar_annotation(EXAMPLE / '1000.json', result['geometry'])
    assert len(gt) == 4
    metrics, _ = summarize(result['boxes'], [item['box'] for item in gt], 0.50)
    assert (metrics['TP'], metrics['FP'], metrics['FN']) == (3, 1, 1)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "examples" / "countable" / "1000.jpg").exists(), reason="AGAR sample image is distributed separately")
def test_batch_exports_one_result(tmp_path):
    folder = tmp_path / 'results'
    info = run_batch(EXAMPLE, folder, max_images=1)
    assert info['processed'] == 1
    assert not info['errors']
    assert (folder / 'detections' / '1000.png').is_file()
    assert (folder / 'evaluations' / '1000.png').is_file()
    assert not (folder / 'baseline').exists()
    with (folder / 'summary_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]['Method'] == 'Improved'
    assert (rows[0]['TP'], rows[0]['FP'], rows[0]['FN']) == ('3', '1', '1')
    with (folder / 'predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        preds = list(csv.DictReader(f))
    assert len(preds) == 4 and 'method' not in preds[0]


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "examples" / "countable" / "1000.jpg").exists(), reason="AGAR sample image is distributed separately")
def test_reviewed_label_export(tmp_path):
    source = cv2.imread(str(EXAMPLE / '1000.jpg'))
    det = detect_colonies(source)
    gt = read_agar_annotation(EXAMPLE / '1000.json', det['geometry'])
    exported = save_reviewed_annotations(tmp_path / 'reviewed' / '1000.json', gt,
                                         det['geometry'], (source.shape[1], source.shape[0]),
                                         {'sample_id': 1001}, '1000')
    assert len(exported['labels']) == 4
    assert exported['annotation_source'] == 'manually_reviewed'


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "examples" / "countable" / "1000.jpg").exists(), reason="AGAR sample image is distributed separately")
def test_unannotated_image_not_scored(tmp_path):
    folder = tmp_path / 'dataset'
    folder.mkdir()
    shutil.copyfile(EXAMPLE / '1000.jpg', folder / 'without_json.jpg')
    out = tmp_path / 'results'
    report = run_batch(folder, out, save_images=False, save_evaluations=False)
    assert report['processed'] == 1 and not report['errors']
    assert report['summary'][0]['Evaluated_images'] == 0
    with (out / 'per_image_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]['has_annotation'] == 'False' and rows[0]['TP'] == ''
