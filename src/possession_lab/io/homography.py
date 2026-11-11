"""Read per-frame pitch-to-image homographies and project normalized pitch coordinates."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from games import DEFAULT_SOURCE, GAME, require
from numpy.typing import NDArray  # pylint: disable=wrong-import-order

MATRIX_COLUMNS = [f"h{row}{col}" for row in range(3) for col in range(3)]


def load_homographies(
    game: str = GAME,
    data_dir: str | os.PathLike[str] | None = None,
    source: str = DEFAULT_SOURCE,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Load one 3×3 homography per video frame, preserving invalid frames as NaN matrices.

    Args:
        game: Competition game key defined by the organizer's ``games.py``.
        data_dir: Optional local root of the licensed data bundle.
        source: Recording source whose camera geometry should be loaded.

    Returns:
        Matrices shaped ``(frames, 3, 3)``, reprojection errors, and boolean validity flags.

    Raises:
        FileNotFoundError: If the homography CSV is absent.
        ValueError: If required matrix or metadata columns are missing.
    """
    path = require(["homography"], game, data_dir, source)[0]
    table = pd.read_csv(path)
    required = ["frame", "reprojection_error", "valid", *MATRIX_COLUMNS]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{path}: missing homography columns {missing}")
    matrices = table[MATRIX_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    errors = table["reprojection_error"].to_numpy(dtype=np.float64)
    valid = table["valid"].to_numpy(dtype=bool)
    return matrices, errors, valid


def project_pitch_to_image(
    matrix: NDArray[np.float64],
    x: float | NDArray[np.float64],
    y: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project normalized pitch coordinates (0–1) to image pixels with a 3×3 matrix.

    Args:
        matrix: Homography for one frame, shaped ``(3, 3)``.
        x: Normalized pitch x-coordinate or array of coordinates.
        y: Normalized pitch y-coordinate or array of coordinates.

    Returns:
        Projected x and y pixel coordinates. Invalid matrix entries propagate as NaN.

    Raises:
        ValueError: If the matrix does not have shape ``(3, 3)``.
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 homography, got {matrix.shape}")
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    denominator = matrix[2, 0] * x_values + matrix[2, 1] * y_values + matrix[2, 2]
    image_x = (matrix[0, 0] * x_values + matrix[0, 1] * y_values + matrix[0, 2]) / denominator
    image_y = (matrix[1, 0] * x_values + matrix[1, 1] * y_values + matrix[1, 2]) / denominator
    return image_x, image_y
