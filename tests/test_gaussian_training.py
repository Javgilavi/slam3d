"""Camera-adapter regressions; optional CUDA trainer dependencies."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('gsplat')
spec = importlib.util.spec_from_file_location('pano_training', Path(__file__).parents[1]/'scripts/train_pano_gaussians.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def test_zenith_and_nadir_optical_axes():
    views = dict(trainer.view_rotations(4, up=True, down=True))
    # Panorama camera coordinates are x right, y down, z forward.
    np.testing.assert_allclose(views['up'] @ [0,0,1], [0,-1,0], atol=1e-10)
    np.testing.assert_allclose(views['down'] @ [0,0,1], [0,1,0], atol=1e-10)
    for R in views.values():
        np.testing.assert_allclose(R.T@R, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1)
