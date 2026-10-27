"""Validated, serializable parameters for the Improved-only colony detector.

Defaults reproduce the reference v2.0 detector (no automatic parameter tuning).
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

IMAGE_SIZE = 1024
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
COLONY_CLASSES = ('S.aureus', 'B.subtilis', 'P.aeruginosa', 'E.coli', 'C.albicans')


@dataclass(frozen=True)
class DetectorConfig:
    """One immutable parameter snapshot for a detection or a complete batch."""
    size: int = IMAGE_SIZE
    dish_shrink: float = 96.0
    gaussian_sigma: float = 1.0
    bg_kernel: int = 121
    l_weight: float = 1.0
    thr_low: float = 3.5
    thr_high: float = 8.0
    multiscale_sigma: float = 2.0
    opening_radius: int = 2
    hole_max: int = 300
    min_area: int = 8
    max_area: int = 25000
    ws_smooth: float = 1.0
    ws_alpha: float = 1.0
    box_pad: int = 1
    edge_r_min: float = 0.90
    edge_r_max: float = 1.09
    edge_h_low: int = 20
    edge_h_high: int = 42
    edge_s_min: int = 80
    edge_v_low: int = 140
    edge_v_high: int = 210
    edge_min_area: int = 250
    edge_max_area: int = 6000
    dedup_iou: float = 0.30
    iou_threshold: float = 0.50

    def __post_init__(self):
        if self.size != IMAGE_SIZE:
            raise ValueError('The current pipeline requires a 1024x1024 working crop')
        bounds = {
            'dish_shrink': (70, 120),
            'gaussian_sigma': (0, 10),
            'bg_kernel': (21, 801),
            'l_weight': (0, 10),
            'thr_low': (0.1, 100),
            'thr_high': (0.1, 200),
            'multiscale_sigma': (0, 20),
            'opening_radius': (0, 20),
            'hole_max': (0, 50000),
            'min_area': (0, 20000),
            'max_area': (50, 1000000),
            'ws_smooth': (0, 10),
            'ws_alpha': (0.1, 10),
            'box_pad': (0, 60),
            'edge_r_min': (0.1, 1.5),
            'edge_r_max': (0.1, 1.5),
            'edge_h_low': (0, 179),
            'edge_h_high': (0, 179),
            'edge_s_min': (0, 255),
            'edge_v_low': (0, 255),
            'edge_v_high': (0, 255),
            'edge_min_area': (1, 50000),
            'edge_max_area': (1, 200000),
            'dedup_iou': (0, 1),
            'iou_threshold': (0, 1),
        }
        for name, (lower, upper) in bounds.items():
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f'{name}: expected a finite value in [{lower}, {upper}], got {value!r}')
        for name in ('opening_radius', 'hole_max', 'min_area', 'max_area', 'box_pad',
                     'bg_kernel', 'edge_h_low', 'edge_h_high', 'edge_s_min', 'edge_v_low',
                     'edge_v_high', 'edge_min_area', 'edge_max_area'):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ValueError(f'{name} must be an integer')
        if self.bg_kernel % 2 != 1:
            raise ValueError('bg_kernel must be odd')
        for lower, upper in (
            ('thr_low', 'thr_high'), ('min_area', 'max_area'),
            ('edge_r_min', 'edge_r_max'), ('edge_h_low', 'edge_h_high'),
            ('edge_v_low', 'edge_v_high'), ('edge_min_area', 'edge_max_area')
        ):
            if getattr(self, lower) > getattr(self, upper):
                raise ValueError(f'{lower} must not exceed {upper}')

    def to_dict(self):
        """Return a plain dictionary suitable for reproducible JSON presets."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Load a preset with strict key/type validation and backward-compatible defaults."""
        if not isinstance(data, dict):
            raise ValueError('Preset must be a JSON object')
        data = data.get('detector', data)
        if not isinstance(data, dict):
            raise ValueError('Preset detector must be a JSON object')
        valid = {field.name: field for field in fields(cls)}
        extra = set(data) - set(valid)
        if extra:
            raise ValueError('Unknown parameter(s): ' + ', '.join(sorted(extra)))
        clean = {}
        for key, value in data.items():
            default_value = getattr(cls(), key)
            if isinstance(default_value, int):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f'{key} must be an integer')
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{key} must be numeric')
            clean[key] = value
        return cls(**clean)


def save_preset(config: DetectorConfig, path):
    """Write the full parameter set for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump({'format_version': 1, 'detector': config.to_dict()}, stream, indent=2)
    return path


def load_preset(path):
    """Validate and load a parameter preset from JSON."""
    with Path(path).open('r', encoding='utf-8') as stream:
        return DetectorConfig.from_dict(json.load(stream))
