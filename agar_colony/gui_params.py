"""Tkinter parameter editor: validates, applies and persists detector presets."""
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import DetectorConfig, load_preset, save_preset

# (parameter, display title, lower, upper, increment)
PARAM_GROUPS = {
    'Dish / Background': [
        ('dish_shrink', 'Dish ROI size (%)', 70, 120, 1),
        ('gaussian_sigma', 'Gaussian sigma', 0, 10, .1),
        ('bg_kernel', 'Background median kernel (odd)', 21, 801, 2),
        ('l_weight', 'Lab lightness weight', 0, 10, .1),
    ],
    'Segmentation': [
        ('thr_low', 'Hysteresis low (MAD units)', .1, 100, .5),
        ('thr_high', 'Hysteresis high (MAD units)', .1, 200, .5),
        ('multiscale_sigma', 'Large-scale sigma', 0, 20, .1),
        ('opening_radius', 'Opening radius (px)', 0, 20, 1),
        ('hole_max', 'Fill holes up to (px²)', 0, 50000, 10),
        ('min_area', 'Minimum area (px²)', 0, 20000, 1),
        ('max_area', 'Maximum area (px²)', 50, 1000000, 50),
    ],
    'Watershed / Boxes': [
        ('ws_smooth', 'Distance smooth sigma', 0, 10, .1),
        ('ws_alpha', 'Marker exclusion multiplier', .1, 10, .1),
        ('box_pad', 'Box padding (px)', 0, 60, 1),
        ('dedup_iou', 'Duplicate suppression IoU', 0, 1, .05),
    ],
    'Edge HSV': [
        ('edge_r_min', 'Annulus inner ratio', .1, 1.5, .01),
        ('edge_r_max', 'Annulus outer ratio', .1, 1.5, .01),
        ('edge_h_low', 'Hue min', 0, 179, 1),
        ('edge_h_high', 'Hue max', 0, 179, 1),
        ('edge_s_min', 'Saturation min', 0, 255, 1),
        ('edge_v_low', 'Value min', 0, 255, 1),
        ('edge_v_high', 'Value max', 0, 255, 1),
        ('edge_min_area', 'Edge region min area', 1, 50000, 10),
        ('edge_max_area', 'Edge region max area', 1, 200000, 10),
    ],
    'Evaluation': [
        ('iou_threshold', 'Ground truth matching IoU', 0, 1, .05),
    ],
}

SLIDER_FIELDS = {'dish_shrink', 'gaussian_sigma', 'thr_low', 'thr_high',
                 'opening_radius', 'ws_smooth', 'ws_alpha', 'edge_h_low',
                 'edge_h_high', 'iou_threshold'}
INTEGER_FIELDS = {
    'bg_kernel', 'opening_radius', 'hole_max', 'min_area', 'max_area',
    'box_pad', 'edge_h_low', 'edge_h_high', 'edge_s_min', 'edge_v_low',
    'edge_v_high', 'edge_min_area', 'edge_max_area', 'size',
}


class ParameterDialog(tk.Toplevel):
    """Non-modal settings window; changed values run only when Apply is pressed."""

    def __init__(self, parent, active_config, on_apply):
        super().__init__(parent)
        self.title('Detector parameters — Improved Only')
        self.geometry('800x590')
        self.minsize(680, 500)
        self.transient(parent)
        self.on_apply = on_apply
        self.vars = {}
        self.sliders = {}
        self._updating = False
        self.feedback_var = tk.StringVar(value='Changes are pending until Apply + Rerun.')
        self._build()
        self.set_config(active_config)
        self.protocol('WM_DELETE_WINDOW', self.destroy)

    def _build(self):
        ttk.Label(self, text='Edit values, then click Apply to re-run the selected image.',
                  font=('Arial', 11, 'bold')).pack(anchor='w', padx=12, pady=(12, 6))
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=12, pady=5)
        self.notebook = notebook

        for group_name, definitions in PARAM_GROUPS.items():
            frame = ttk.Frame(notebook, padding=14)
            notebook.add(frame, text=group_name)
            frame.columnconfigure(2, weight=1)
            for row, (name, label, low, high, step) in enumerate(definitions):
                ttk.Label(frame, text=label, width=34).grid(
                    row=row, column=0, sticky='w', padx=(0, 8), pady=7)
                var = tk.StringVar()
                self.vars[name] = var
                control = ttk.Spinbox(frame, from_=low, to=high, increment=step,
                                      textvariable=var, width=12)
                control.grid(row=row, column=1, sticky='w', padx=(0, 16))
                if name in SLIDER_FIELDS:
                    slider = ttk.Scale(frame, from_=low, to=high, orient='horizontal',
                                       command=lambda v, key=name, inc=step:
                                       self._from_slider(key, float(v), inc))
                    slider.grid(row=row, column=2, sticky='ew', padx=8)
                    self.sliders[name] = (slider, low, high)
                    # Commit precise values entered by keyboard on focus loss or Enter.
                    control.bind('<Return>', lambda _event, key=name: self._sync_slider(key))
                    control.bind('<FocusOut>', lambda _event, key=name: self._sync_slider(key))

        ttk.Label(self, text='Presets store detection and evaluation parameters. '
                  'Ground-truth JSON is never used to detect colonies.',
                  foreground='#64748b', wraplength=750).pack(anchor='w', padx=12, pady=(5, 1))
        ttk.Label(self, textvariable=self.feedback_var, foreground='#225c8a').pack(
            anchor='w', padx=12, pady=(0, 4))

        actions = ttk.Frame(self)
        actions.pack(fill='x', padx=12, pady=(4, 14))
        ttk.Button(actions, text='Reset Default', command=self.reset_defaults).pack(side='left', padx=3)
        ttk.Button(actions, text='Load Preset', command=self.load).pack(side='left', padx=3)
        ttk.Button(actions, text='Save Preset', command=self.save).pack(side='left', padx=3)
        ttk.Button(actions, text='Close', command=self.destroy).pack(side='right', padx=3)
        ttk.Button(actions, text='Apply + Rerun', command=self.apply).pack(side='right', padx=3)

    def _from_slider(self, key, value, step):
        if self._updating:
            return
        # Round the slider's output to readable increments; exact values can be typed.
        snapped = round(value / step) * step
        self.vars[key].set(str(int(round(snapped))) if key in INTEGER_FIELDS
                           else f'{snapped:.3f}'.rstrip('0').rstrip('.'))

    def _sync_slider(self, key):
        if key not in self.sliders:
            return
        try:
            value = float(self.vars[key].get())
        except ValueError:
            return
        slider, low, high = self.sliders[key]
        self._updating = True
        slider.set(min(high, max(low, value)))
        self._updating = False

    def set_config(self, config):
        self._updating = True
        for name, var in self.vars.items():
            var.set(str(getattr(config, name)))
            if name in self.sliders:
                slider, _low, _high = self.sliders[name]
                slider.set(getattr(config, name))
        self._updating = False

    def parse_config(self):
        """Parse all fields together; do not change active settings on an invalid value."""
        data = {'size': DetectorConfig().size}
        for name, var in self.vars.items():
            value = var.get().strip().replace(',', '.')
            try:
                if name in INTEGER_FIELDS:
                    # Reject fractional values instead of silently truncating them.
                    numeric = float(value)
                    if not numeric.is_integer():
                        raise ValueError(f'{name} must be an integer')
                    data[name] = int(numeric)
                else:
                    data[name] = float(value)
            except ValueError as exc:
                raise ValueError(f'Invalid {name}: {value!r} ({exc})') from exc
        return DetectorConfig.from_dict(data)

    def apply(self, silent=False):
        try:
            config = self.parse_config()
        except ValueError as exc:
            messagebox.showerror('Invalid detector parameter', str(exc), parent=self)
            return False
        if not self.on_apply(config):
            return False
        self.feedback_var.set('Applied. The selected image is reprocessing; batch will use this preset.')
        return True

    def reset_defaults(self):
        self.set_config(DetectorConfig())
        self.notebook.select(0)
        self.feedback_var.set('Default values restored in editor. Click Apply + Rerun to activate.')

    def save(self):
        try:
            config = self.parse_config()
        except ValueError as exc:
            messagebox.showerror('Invalid detector parameter', str(exc), parent=self)
            return
        path = filedialog.asksaveasfilename(parent=self, title='Save parameter preset',
                                            defaultextension='.json',
                                            initialfile='agar_parameters.json',
                                            filetypes=[('JSON files', '*.json')])
        if not path:
            return
        try:
            save_preset(config, path)
        except OSError as exc:
            messagebox.showerror('Save failed', str(exc), parent=self)
            return
        self.feedback_var.set('Preset saved. Click Apply + Rerun to activate these values.')

    def load(self):
        path = filedialog.askopenfilename(parent=self, title='Load parameter preset',
                                          filetypes=[('JSON files', '*.json')])
        if not path:
            return
        try:
            config = load_preset(path)
        except (ValueError, OSError) as exc:
            messagebox.showerror('Invalid preset', str(exc), parent=self)
            return
        self.set_config(config)
        self.feedback_var.set('Preset loaded. Click Apply + Rerun to activate it.')
