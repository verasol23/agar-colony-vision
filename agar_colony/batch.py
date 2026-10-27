"""Dataset batch detection with one Improved result per image.

The detector is independent from JSON. Ground truth is only used for scoring.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from threading import Event
from typing import Callable
import cv2
from .annotations import read_agar_annotation
from .config import IMAGE_EXTENSIONS, DetectorConfig, save_preset
from .detection import detect_colonies
from .evaluation import summarize
from .imaging import render_detection, render_evaluation
from .logging_utils import setup_file_logging

PRED_COLUMNS = ['image', 'id', 'class', 'source', 'x1', 'y1', 'x2', 'y2']
IMAGE_COLUMNS = ['image', 'has_annotation', 'dish_method',
                 'GT', 'Pred', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
SUMMARY_COLUMNS = ['Method', 'Evaluated_images', 'GT', 'Pred', 'TP', 'FP', 'FN',
                   'Precision', 'Recall', 'F1']
ERROR_COLUMNS = ['image', 'stage', 'error']


def list_images(dataset_dir):
    """Find all supported images recursively in a reproducible order."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(path for path in root.rglob('*')
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _save_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def summary_from_rows(rows):
    """Micro-average TP, FP and FN using annotated images only."""
    scored = [row for row in rows if row['has_annotation']]
    tp = sum(row['TP'] for row in scored)
    fp = sum(row['FP'] for row in scored)
    fn = sum(row['FN'] for row in scored)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return [dict(Method='Improved', Evaluated_images=len(scored), GT=tp + fn,
                 Pred=tp + fp, TP=tp, FP=fp, FN=fn,
                 Precision=precision, Recall=recall, F1=f1)]


def run_batch(dataset_dir, output_dir, config=None, max_images=None,
              save_images=True, save_evaluations=True, save_masks=False,
              progress: Callable | None = None, stop_event: Event | None = None):
    """Run Improved detection on all images; preserve originals and report failures."""
    config = config or DetectorConfig()
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if dataset_dir == output_dir or dataset_dir in output_dir.parents:
        raise ValueError('Output directory must be outside the dataset directory')

    images = list_images(dataset_dir)
    if max_images is not None:
        if max_images < 0:
            raise ValueError('max_images must be >= 0')
        images = images[:max_images]

    output_dir.mkdir(parents=True, exist_ok=True)
    save_preset(config, output_dir / 'config_used.json')
    logger = setup_file_logging(output_dir / 'logs' / 'batch.log')
    logger.info('Start improved-only dataset=%s, images=%d, iou=%.2f',
                dataset_dir, len(images), config.iou_threshold)
    predictions, image_metrics, errors = [], [], []
    completed = 0

    for image_path in images:
        if stop_event is not None and stop_event.is_set():
            break
        relative = image_path.relative_to(dataset_dir)
        name = relative.as_posix()
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise OSError('Could not load image')
            detection = detect_colonies(image, config)
            boxes = detection['boxes']
            json_path = image_path.with_suffix('.json')
            gt = None
            try:
                gt = read_agar_annotation(json_path, detection['geometry'])
            except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
                errors.append(dict(image=name, stage='annotation', error=str(exc)))
                logger.warning('JSON error %s: %s', name, exc)

            if gt is None:
                stats = dict(GT=None, Pred=len(boxes), TP=None, FP=None, FN=None,
                             Precision=None, Recall=None, F1=None)
                matched = None
                truth_boxes = []
            else:
                truth_boxes = [item['box'] for item in gt]
                stats, matched = summarize(boxes, truth_boxes, config.iou_threshold)

            image_metrics.append(dict(image=name, has_annotation=gt is not None,
                                      dish_method=detection['dish_method'], **stats))
            for idx, box in enumerate(boxes):
                provenance = 'inner' if idx < detection['inner_count'] else 'edge'
                predictions.append(dict(image=name, id=idx + 1, **{'class': 'unclassified'},
                                        source=provenance, x1=box[0], y1=box[1],
                                        x2=box[2], y2=box[3]))
            if save_images:
                target = output_dir / 'detections' / relative.with_suffix('.png')
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), render_detection(detection['crop'], boxes)):
                    raise OSError(f'Could not save {target}')
            if save_evaluations and matched is not None:
                target = output_dir / 'evaluations' / relative.with_suffix('.png')
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target), render_evaluation(
                        detection['crop'], boxes, truth_boxes, matched)):
                    raise OSError(f'Could not save {target}')
            if save_masks:
                for stage in ('mask', 'hysteresis', 'morphology', 'edge_mask'):
                    target = output_dir / 'masks' / stage / relative.with_suffix('.png')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(target), detection[stage]):
                        raise OSError(f'Could not save {target}')

            status = 'ok'
            logger.info('OK %s boxes=%d inner=%d edge_added=%d json=%s',
                        name, len(boxes), detection['inner_count'],
                        detection['edge_added'], gt is not None)
        except Exception as exc:
            status = 'error'
            errors.append(dict(image=name, stage='detection',
                               error=f'{type(exc).__name__}: {exc}'))
            logger.exception('Error processing %s', name)
        completed += 1
        if progress is not None:
            progress(completed, len(images), name, status)

    summary = summary_from_rows(image_metrics)
    _save_csv(output_dir / 'predictions.csv', predictions, PRED_COLUMNS)
    _save_csv(output_dir / 'per_image_metrics.csv', image_metrics, IMAGE_COLUMNS)
    _save_csv(output_dir / 'summary_metrics.csv', summary, SUMMARY_COLUMNS)
    _save_csv(output_dir / 'errors.csv', errors, ERROR_COLUMNS)
    logger.info('Done %d/%d; errors=%d', completed, len(images), len(errors))
    return dict(total=len(images), processed=completed,
                cancelled=completed < len(images), summary=summary,
                errors=errors, output=output_dir)
