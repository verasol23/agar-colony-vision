"""Load and export AGAR JSON and YOLO annotations without mutating source images."""
import json
from pathlib import Path
from .config import COLONY_CLASSES, IMAGE_SIZE as SIZE
from .dish import original_to_work

def read_agar_annotation(json_path, geom):
    """Load optional JSON ground truth and map boxes to cropped-image coordinates."""
    if not json_path.is_file():
        return None

    with json_path.open('r', encoding='utf-8-sig') as stream:
        data = json.load(stream)

    gt = []
    for item in data.get('labels', []):
        try:
            x, y = int(item['x']), int(item['y'])
            width, height = int(item['width']), int(item['height'])
            if width <= 0 or height <= 0:
                continue
            box = original_to_work([x, y, x + width, y + height], geom)
            if box[2] > box[0] and box[3] > box[1]:
                gt.append({'class': str(item.get('class', '')), 'box': box})
        except (KeyError, TypeError, ValueError):
            continue
    return gt


def work_to_original(box, geom, original_size=None):
    """Convert work-space xyxy boxes to the original image coordinate system."""
    x1, y1, x2, y2 = box
    vals = [(x1 / geom['scale'] + geom['x0']),
            (y1 / geom['scale'] + geom['y0']),
            (x2 / geom['scale'] + geom['x0']),
            (y2 / geom['scale'] + geom['y0'])]
    rounded = [int(round(v)) for v in vals]
    if original_size is not None:
        width, height = original_size
        rounded = [max(0, min(rounded[i], width if i % 2 == 0 else height)) for i in range(4)]
    return rounded


def save_reviewed_annotations(destination, annotated_boxes, geom, original_size,
                              metadata=None, image_name=None):
    """Write manually reviewed boxes in original-image AGAR JSON coordinates.

    This function never modifies the original dataset annotation.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for index, item in enumerate(annotated_boxes, 1):
        class_name = str(item.get('class', 'unclassified'))
        if class_name not in COLONY_CLASSES:
            raise ValueError(f'Cannot export unsupported class label: {class_name}')
        x1, y1, x2, y2 = work_to_original(item['box'], geom, original_size)
        if x2 <= x1 or y2 <= y1:
            continue
        items.append({'id': index, 'class': class_name, 'x': x1, 'y': y1,
                      'width': x2-x1, 'height': y2-y1})
    data = {
        'background': (metadata or {}).get('background', 'bright'),
        'classes': sorted({x['class'] for x in items}),
        'colonies_number': len(items),
        'labels': items,
        'sample_id': (metadata or {}).get('sample_id', Path(image_name or destination.stem).stem),
        'annotation_source': 'manually_reviewed',
    }
    destination.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return data


def write_yolo_annotations(destination, annotated_boxes):
    """Save manually assigned class labels in normalized YOLO xywh coordinates."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in annotated_boxes:
        class_name = str(item.get('class', 'unclassified'))
        if class_name not in COLONY_CLASSES:
            raise ValueError(f'Unsupported class label: {class_name}')
        x1,y1,x2,y2 = item['box']
        if not (0 <= x1 < x2 <= SIZE and 0 <= y1 < y2 <= SIZE):
            continue
        cx = (x1+x2)/(2*SIZE)
        cy = (y1+y2)/(2*SIZE)
        w = (x2-x1)/SIZE
        h = (y2-y1)/SIZE
        lines.append(f'{COLONY_CLASSES.index(class_name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
    destination.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    return len(lines)
