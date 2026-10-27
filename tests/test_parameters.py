"""Parameter validation, persistence and real detector propagation tests."""
import csv
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
import pytest

from agar_colony.config import DetectorConfig, load_preset, save_preset
from agar_colony.detection import detect_colonies
from agar_colony.batch import run_batch

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'countable'


def test_preset_roundtrip_and_defaults(tmp_path):
    config = replace(DetectorConfig(), gaussian_sigma=1.5, thr_low=4.0,
                     thr_high=9.0, opening_radius=1, ws_smooth=1.5,
                     edge_h_low=19, iou_threshold=.65)
    path = save_preset(config, tmp_path / 'configs' / 'experiment.json')
    assert path.is_file()
    assert load_preset(path) == config
    assert DetectorConfig.from_dict({'thr_low': 4.0}) == replace(DetectorConfig(), thr_low=4.0)


@pytest.mark.parametrize('invalid', [
    {'thr_low': 12, 'thr_high': 8}, {'opening_radius': -1},
    {'edge_h_low': 80, 'edge_h_high': 30}, {'bg_kernel': 120},
    {'iou_threshold': 1.1}, {'ws_alpha': float('nan')},
    {'opening_radius': 1.5}, {'not_a_parameter': 3},
])
def test_invalid_config_rejected(invalid):
    with pytest.raises(ValueError):
        DetectorConfig.from_dict(invalid)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "examples" / "countable" / "1000.jpg").exists(), reason="AGAR sample image is distributed separately")
def test_tuning_parameters_affect_real_images():
    image = cv2.imread(str(EXAMPLE / '1000.jpg'))
    base = detect_colonies(image, DetectorConfig())
    changed = detect_colonies(image, replace(DetectorConfig(),
                                           thr_low=12, thr_high=22,
                                           opening_radius=3))
    assert not np.array_equal(base['morphology'], changed['morphology'])
    assert base['boxes'] != changed['boxes']


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "examples" / "countable" / "1000.jpg").exists(), reason="AGAR sample image is distributed separately")
def test_batch_records_applied_config(tmp_path):
    config = replace(DetectorConfig(), gaussian_sigma=1.4,
                     iou_threshold=0.35, opening_radius=1)
    info = run_batch(EXAMPLE, tmp_path / 'results', config, max_images=1,
                     save_images=False, save_evaluations=False)
    assert info['processed'] == 1 and not info['errors']
    assert load_preset(tmp_path / 'results' / 'config_used.json') == config
    with (tmp_path / 'results' / 'per_image_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['has_annotation'] == 'True'
