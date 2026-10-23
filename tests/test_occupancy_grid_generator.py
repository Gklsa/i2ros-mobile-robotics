import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ad_perception"))

from ad_perception.occupancy_grid_generator import OccupancyGridGenerator


def test_build_grid_ignores_out_of_bounds_cells(monkeypatch):
    node = OccupancyGridGenerator.__new__(OccupancyGridGenerator)
    node._forward_range = 30.0
    node._lateral_range = 16.0
    node._resolution = 0.25
    node._pixel_stride = 1
    node._mark_free_space = False
    node._max_depth = 40.0
    node._fallback_fov_deg = 60.0
    node._warned_fallback_info = False
    node._camera_info = None

    def fake_intrinsics():
        return 1.0, 1.0, 0.0, 0.0

    monkeypatch.setattr(node, "_camera_intrinsics", fake_intrinsics)
    monkeypatch.setattr(node, "_point_to_cell", lambda forward, lateral: (0, 64))

    depth = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    grid = node._build_grid(depth, "32FC1")

    assert grid.info.width > 0
    assert grid.info.height > 0
