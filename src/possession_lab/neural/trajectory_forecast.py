"""Physics-based confidence-aware ball trajectory forecasting and completion (N019).

In tracking datasets, ball occlusions and dropouts create missingness. Rather than relying solely
on constant forward/backward filling or unconstrained neural generative heads
(which failed in N014/N017),
this module provides:
1. Physically constrained C1-continuous Hermite spline interpolation across occluded spans.
2. Damped ballistic constant-velocity extrapolation for open-ended dropouts.
3. Explicit exponential confidence gating c_t in [0, 1] reflecting temporal uncertainty.
4. Benchmark utilities to quantify imputation error against baseline forward-filling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]
HALF_LENGTH = 52.5
HALF_WIDTH = 34.0


def _central_diff_velocity(pos: F32, seen: NDArray[np.bool_], fps: float = 25.0) -> F32:
    """Velocity computed across observed frames; zero where unobserved."""
    t_len = len(pos)
    vel = np.zeros_like(pos)
    for i in range(t_len):
        if not seen[i]:
            continue
        prev_idx = i - 1
        while prev_idx >= 0 and not seen[prev_idx]:
            prev_idx -= 1
        next_idx = i + 1
        while next_idx < t_len and not seen[next_idx]:
            next_idx += 1
        if prev_idx >= 0 and next_idx < t_len:
            dt = (next_idx - prev_idx) / fps
            vel[i] = (pos[next_idx] - pos[prev_idx]) / max(dt, 1e-3)
        elif next_idx < t_len:
            dt = (next_idx - i) / fps
            vel[i] = (pos[next_idx] - pos[i]) / max(dt, 1e-3)
        elif prev_idx >= 0:
            dt = (i - prev_idx) / fps
            vel[i] = (pos[i] - pos[prev_idx]) / max(dt, 1e-3)
    return np.clip(vel, -30.0, 30.0)


def forecast_ball_trajectory(
    ball_xy: F32,
    seen: NDArray[np.bool_],
    fps: float = 25.0,
    decay_tau_s: float = 0.6,
    damping_gamma: float = 0.08,
) -> tuple[F32, F32, F32]:
    """Compute confidence-aware physical trajectory forecast for ball coordinates.

    Args:
        ball_xy: ``[T, 2]`` pitch coordinates in metres (may contain NaNs where missing).
        seen: ``[T]`` boolean mask indicating valid ball observations.
        fps: Sampling rate (default 25.0 Hz).
        decay_tau_s: Confidence decay time in seconds (default 0.6 s = 15 frames).
        damping_gamma: Velocity damping parameter for extrapolation (air/grass friction).

    Returns:
        tuple of:
            forecast_xy: ``[T, 2]`` forecasted/interpolated pitch coordinates in metres.
            forecast_conf: ``[T, 1]`` confidence score in [0.0, 1.0].
            forecast_imputed: ``[T, 1]`` indicator (1.0 = imputed, 0.0 = observed).
    """
    # The explicit model/state-machine structure keeps the domain cases auditable.
    # pylint: disable=too-many-statements
    t_len = len(ball_xy)
    clean_xy = np.nan_to_num(ball_xy).astype(np.float32)
    forecast_xy = clean_xy.copy()
    forecast_conf = np.ones((t_len, 1), dtype=np.float32)
    forecast_imputed = (~seen[:, None]).astype(np.float32)

    if np.all(seen) or not np.any(seen):
        if not np.any(seen):
            forecast_conf.fill(0.0)
            forecast_xy.fill(0.0)
        return forecast_xy, forecast_conf, forecast_imputed

    vel = _central_diff_velocity(clean_xy, seen, fps=fps)
    tau_frames = max(1.0, decay_tau_s * fps)
    extrap_tau_frames = max(1.0, tau_frames * 0.5)

    # Identify contiguous missing intervals [start, end)
    missing = ~seen
    diff = np.diff(np.pad(missing.astype(np.int8), (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends, strict=True):
        length = e - s
        has_left = s > 0 and seen[s - 1]
        has_right = e < t_len and seen[e]

        if has_left and has_right:
            # Interpolated span: Cubic Hermite spline between left (s-1) and right (e)
            p0 = clean_xy[s - 1]
            p1 = clean_xy[e]
            v0 = vel[s - 1]
            v1 = vel[e]
            dt = (length + 1) / fps

            # u in (0, 1) across the missing gap
            steps = np.arange(1, length + 1, dtype=np.float32) / (length + 1)
            u = steps[:, None]
            u2 = u * u
            u3 = u2 * u

            h00 = 2.0 * u3 - 3.0 * u2 + 1.0
            h10 = u3 - 2.0 * u2 + u
            h01 = -2.0 * u3 + 3.0 * u2
            h11 = u3 - u2

            interp = (
                h00 * p0[None, :]
                + h10 * (v0[None, :] * dt)
                + h01 * p1[None, :]
                + h11 * (v1[None, :] * dt)
            )

            # Clamp to pitch boundaries
            interp[:, 0] = np.clip(interp[:, 0], -HALF_LENGTH, HALF_LENGTH)
            interp[:, 1] = np.clip(interp[:, 1], -HALF_WIDTH, HALF_WIDTH)
            forecast_xy[s:e] = interp

            # Confidence decays as distance to nearest endpoint increases
            dist_to_edge = np.minimum(np.arange(1, length + 1), np.arange(length, 0, -1))
            conf = np.exp(-dist_to_edge / tau_frames)
            forecast_conf[s:e, 0] = conf.astype(np.float32)

        elif has_left:
            # Extrapolation forward from left boundary
            p0 = clean_xy[s - 1]
            v0 = vel[s - 1]
            k = np.arange(1, length + 1, dtype=np.float32)
            dt = k / fps
            damped_dt = (1.0 - np.exp(-damping_gamma * dt)) / max(damping_gamma, 1e-4)
            extrap = p0[None, :] + v0[None, :] * damped_dt[:, None]
            extrap[:, 0] = np.clip(extrap[:, 0], -HALF_LENGTH, HALF_LENGTH)
            extrap[:, 1] = np.clip(extrap[:, 1], -HALF_WIDTH, HALF_WIDTH)
            forecast_xy[s:e] = extrap

            conf = np.exp(-k / extrap_tau_frames)
            forecast_conf[s:e, 0] = conf.astype(np.float32)

        elif has_right:
            # Backward extrapolation from right boundary
            p1 = clean_xy[e]
            v1 = vel[e]
            k = np.arange(length, 0, -1, dtype=np.float32)
            dt = k / fps
            damped_dt = (1.0 - np.exp(-damping_gamma * dt)) / max(damping_gamma, 1e-4)
            extrap = p1[None, :] - v1[None, :] * damped_dt[:, None]
            extrap[:, 0] = np.clip(extrap[:, 0], -HALF_LENGTH, HALF_LENGTH)
            extrap[:, 1] = np.clip(extrap[:, 1], -HALF_WIDTH, HALF_WIDTH)
            forecast_xy[s:e] = extrap

            conf = np.exp(-k / extrap_tau_frames)
            forecast_conf[s:e, 0] = conf.astype(np.float32)

    return forecast_xy, forecast_conf, forecast_imputed


def benchmark_trajectory_completion(
    clean_xy: F32,
    gap_lengths: tuple[int, ...] = (5, 10, 25, 50),
    n_trials: int = 40,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Benchmark trajectory forecast vs constant fill across synthetic occlusion gaps.

    Evaluates MAE and RMSE in metres on external tracking data.

    Returns:
        Dict mapping gap length to fill and forecast MAE, RMSE, and percentage gain.
    """
    rng = np.random.default_rng(seed)
    valid_indices = np.where(~np.isnan(clean_xy[:, 0]))[0]
    results: dict[str, dict[str, float]] = {}

    for gap in gap_lengths:
        if len(valid_indices) < gap + 2:
            continue
        fill_errors: list[float] = []
        forecast_errors: list[float] = []

        for _ in range(n_trials):
            # Pick a contiguous slice that is observed
            idx = int(rng.choice(valid_indices[:-gap]))
            if np.any(np.isnan(clean_xy[idx : idx + gap, 0])):
                continue
            ground_truth = clean_xy[idx : idx + gap]

            # Construct synthetic mask with this gap missing
            sim_seen = np.ones(len(clean_xy), dtype=bool)
            sim_seen[idx : idx + gap] = False
            sim_xy = clean_xy.copy()
            sim_xy[idx : idx + gap] = np.nan

            # 1. Baseline forward-fill
            last_known = clean_xy[idx - 1] if idx > 0 else clean_xy[idx + gap]
            fill_pred = np.tile(last_known, (gap, 1))

            # 2. Physics trajectory forecast
            fc_xy, _, _ = forecast_ball_trajectory(sim_xy, sim_seen)
            fc_pred = fc_xy[idx : idx + gap]

            fill_err = np.linalg.norm(ground_truth - fill_pred, axis=-1)
            fc_err = np.linalg.norm(ground_truth - fc_pred, axis=-1)

            fill_errors.extend(fill_err.tolist())
            forecast_errors.extend(fc_err.tolist())

        if fill_errors:
            f_mae = float(np.mean(fill_errors))
            fc_mae = float(np.mean(forecast_errors))
            f_rmse = float(np.sqrt(np.mean(np.square(fill_errors))))
            fc_rmse = float(np.sqrt(np.mean(np.square(forecast_errors))))
            gain = ((f_mae - fc_mae) / max(f_mae, 1e-4)) * 100.0
            results[str(gap)] = {
                "fill_mae": f_mae,
                "forecast_mae": fc_mae,
                "fill_rmse": f_rmse,
                "forecast_rmse": fc_rmse,
                "gain_pct": gain,
            }

    return results
