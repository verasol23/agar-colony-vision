# AGAR Colony Vision

**Classical computer vision for Petri-dish colony detection and human-in-the-loop bounding-box annotation.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-classical%20vision-green) ![Tests](https://img.shields.io/badge/tests-pytest-informational)

This repository contains **AGAR Colony Vision v2.1 (Improved-Only)**, a modular Python application for detecting and reviewing microbial-colony locations in Petri-dish photographs. It uses classical image processing (not a trained YOLO/CNN detector) and provides both a Tkinter GUI and a batch CLI.

The project was developed as an academic machine-vision project in **September–October 2026**.

## Highlights

- **Petri-dish localization:** grayscale thresholding/contours, circular crop, and elliptical region-of-interest mask.
- **Colony candidate detection:** Lab color difference against estimated agar background, hysteresis thresholds, morphology, and watershed separation.
- **Near-rim recovery:** a complementary HSV-based annulus branch recovers possible colonies around the plate edge.
- **Human-in-the-loop annotation:** inspect and edit bounding boxes, manually assign a class, and export reviewed annotations as AGAR JSON and YOLO-style labels.
- **Traceable evaluation:** compare detections with paired JSON ground truth using IoU and TP/FP/FN metrics; export per-image and aggregate CSV reports.
- **Adjustable parameters:** GUI settings and reproducible JSON presets for detector thresholds, morphology, watershed, and IoU evaluation.

> **Important scientific limitation:** Automatic predictions are **unclassified**. The application does **not** automatically identify bacterial or fungal species. Species names in reviewed annotations are selected by a human. IoU-based F1 measures **colony localization**, not species-classification accuracy.

## Pipeline

```text
Input dish image
   |
   +--> Petri-dish detection / 1024x1024 crop / ROI mask
   |
   +--> Background estimation + Lab color difference
   |        |
   |        +--> Hysteresis --> Morphology --> Watershed --> Inner colonies
   |
   +--> HSV ring analysis -------------------------------> Edge colonies
                                                               |
                        Merge + deduplicate boxes <------------+
                                      |
                       Manual review + JSON/YOLO export
                                      |
                         Optional IoU/TP/FP/FN evaluation
```

For stage-by-stage explanation, see [docs/PIPELINE.md](docs/PIPELINE.md).

## Quick start

Requires **Python 3.10+**. On Ubuntu/Debian, Tkinter may need to be installed with `sudo apt install python3-tk`.

```bash
# Clone and install
git clone https://github.com/verasol23/agar-colony-vision.git
cd agar-colony-vision
python -m venv .venv
source .venv/bin/activate             # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Launch the graphical interface
python main.py
```

In the GUI, choose a folder of dish images, open an image, inspect processing stages, adjust parameters using **Parameters...**, and draw/edit bounding boxes. Select a species label **manually** before exporting reviewed annotations.

### Batch processing (CLI)

```bash
python -m agar_colony.cli \
  --dataset /path/to/your/countable \
  --output results \
  --preset presets/default.json \
  --iou 0.50
```

Use `--limit 10` for a small batch, `--save-masks` for debugging masks, or `--no-images` to generate only reports. Keep the `--output` folder **outside** the input dataset directory.

### Expected data layout

Images can be processed without reference annotations. For evaluation, pair each image with a matching AGAR JSON ground-truth file, for example:

```text
your-dataset/
  countable/
    sample_001.jpg
    sample_001.json     # optional; used for review/evaluation, not detection
    sample_002.jpg
```

**Dataset licensing:** The original sample photograph and screenshots were omitted from this public repository until redistribution rights are confirmed. The code supports your own images. Image-dependent regression tests automatically skip when the historical sample `examples/countable/1000.jpg` is absent.

## Repository layout

```text
agar_colony/
  dish.py                # Dish localization and crop
  preprocessing.py       # Agar/background preparation
  segmentation.py        # Hysteresis, morphology, watershed
  edge.py                # HSV annulus candidate branch
  detection.py           # Unified detector pipeline
  annotations.py         # AGAR JSON and reviewed YOLO export
  evaluation.py          # IoU, TP, FP, FN
  batch.py               # Folder processing and CSV metrics
  gui.py                 # Tkinter GUI
  gui_params.py          # Parameter editor / preset management
  gui_panels.py          # Interactive image/box widgets
  cli.py                 # Command-line interface
docs/                     # Pipeline, parameter, and report explanations
presets/default.json      # Reproducible default settings
tests/                    # Unit and integration tests
main.py                   # GUI entry point
requirements.txt
requirements-dev.txt
```

## Evaluation and reproducibility

To run the included tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

A **historical single-image demo**, when run with the original `1000.jpg` and paired JSON under the v2.1 preset, gave **4 predicted boxes, 4 GT boxes, TP=3, FP=1, FN=1, F1=0.75 at IoU 0.50**. **This is not a full-dataset benchmark**. The corrected v2.1 detection pipeline has not been validated across the complete 120-image set.

Exported reports include `predictions.csv`, `per_image_metrics.csv`, `summary_metrics.csv`, `errors.csv`, and `config_used.json`. Ground-truth JSON is not used by the detector, only by review/evaluation logic.

## Limitations

- Lighting, glare, printed dish markings, near-rim colonies, and overlapping colonies may reduce detection reliability.
- Detector settings can require adjustment to fit a different camera or culture medium.
- Species labels are manual; automated species identification requires an independent validated classification pipeline.
- The included tests and single-image demonstration are not a substitute for evaluation on a representative held-out dataset.

## Documentation

- [Pipeline & debug reference](docs/PIPELINE.md)
- [Parameters and presets](docs/PARAMETERS.md)
- [Report guide (Vietnamese)](docs/REPORT_GUIDE.md)

## Author

**Tran Trang Sang** — Robotics & Artificial Intelligence Engineering, HCM-UTE  
[GitHub](https://github.com/verasol23)

> No license has been specified in this repository. Please obtain permission before redistributing third-party datasets or images.
