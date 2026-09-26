# Contributing

Contributions that improve reproducibility, documentation, geometry checks, and viewer usability are welcome.

## Set up

Follow the [README quick start](README.md#quick-start) for a full run. For source-only CPU work, install Python 3.12 and use `python -m pip install -e ".[dev]"`. Run the focused CPU suite with:

```bash
python -m pytest -q tests/test_geometry.py tests/test_scale_stitch.py tests/test_align_pf.py
```

The Gaussian tests additionally require PyTorch. End-to-end viewer checks require a processed run and Playwright Chromium; full reconstruction requires the hardware and downloaded assets described in the README.

## Pull requests

Explain the behavior changed, any coordinate-frame or unit assumptions, and how you verified it. Include before/after screenshots for viewer changes. Keep downloaded datasets, model weights, virtual environments, and generated `outputs/` out of commits. When reporting benchmark improvements, state the sequence, evaluation split, hardware, and whether ground truth entered the method or only the evaluation.

The public Hilti data and screenshots have their own [CC BY-NC-SA 3.0 terms](README.md#data-and-image-credits). Upstream dependencies retain their own licenses.
