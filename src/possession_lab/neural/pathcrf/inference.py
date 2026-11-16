"""Full-period, no-GT inference and multi-rate sequence upsampling for N020 PathCRF.

Performs:
1. Period-aware windowed inference with ramp overlap blending.
2. Model-predicted Dead-ball gating and path evidence fusion (no oracle episode boundaries).
3. Exact temporal upsampling to the challenge's 25Hz frame count.
4. Independent Viterbi decoding per period, preserving X on official gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from possession_lab.neural.canonical import CanonicalMatch
from possession_lab.neural.decode import DecodeConfig, decode_match

from .adapter import PathCRFBatch, canonical_to_pathcrf
from .backbone import PathCRFNet
from .fusion import fuse_n020_probabilities

CPU_DEVICE = torch.device("cpu")


@dataclass
class PeriodLogits:
    """Model logits and sample positions for one canonical match period."""

    p_start: int
    p_end: int
    n_p_frames: int
    sampled_frame_ids: NDArray[Any]
    dead_logits: torch.Tensor
    team_logits: torch.Tensor
    edge_logits: torch.Tensor


def _window_ramp_weights(length: int, overlap: int) -> NDArray[Any]:
    """Construct a trapezoidal weighting window for smooth blending across overlaps."""
    weights = np.ones(length, dtype=np.float32)
    if overlap > 0 and length > 2 * overlap:
        ramp_up = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        ramp_down = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)
        weights[:overlap] = ramp_up
        weights[-overlap:] = ramp_down
    return weights


@torch.no_grad()  # type: ignore[untyped-decorator]
def extract_n020_match_logits(
    model: PathCRFNet,
    match: CanonicalMatch,
    *,
    sample_fps: float = 25.0,
    device: torch.device = CPU_DEVICE,
    window_seconds: float = 10.0,
    overlap_seconds: float = 2.0,
    precomputed_batch: PathCRFBatch | None = None,
) -> list[PeriodLogits]:
    """Compute blended raw multi-task logits per period without applying path_weight."""
    # The explicit model/state-machine structure keeps the domain cases auditable.
    # pylint: disable=too-many-locals,too-many-statements
    model.eval()
    model.to(device)

    total_frames = match.n_frames
    if precomputed_batch is not None:
        full_batch = precomputed_batch
    else:
        full_batch = canonical_to_pathcrf(
            match,
            lo=0,
            hi=total_frames,
            target_fps=sample_fps,
            include_labels=False,
        )

    x_full = full_batch.x[0].to(device)  # [T_s_total, 26, F]
    valid_full = full_batch.node_valid[0].to(device)  # [T_s_total, 26]
    sampled_frame_ids_full = full_batch.source_frame_ids[0].numpy()  # [T_s_total]

    periods = np.unique(match.period)
    window_len_samples = max(4, int(round(window_seconds * sample_fps)))
    stride_samples = max(1, int(round((window_seconds - overlap_seconds) * sample_fps)))
    overlap_samples = window_len_samples - stride_samples
    ramp_w_full = _window_ramp_weights(window_len_samples, overlap_samples)
    sample_step = max(1, int(round(25.0 / sample_fps)))

    period_results = []
    for p in periods:
        if p == 0:
            continue

        p_idx = np.flatnonzero(match.period == p)
        if len(p_idx) == 0:
            continue

        p_start = int(p_idx[0])
        p_end = int(p_idx[-1]) + 1
        n_p_frames = p_end - p_start

        p_s_start = p_start // sample_step
        p_s_end = min(len(x_full), int(np.ceil(p_end / sample_step)))
        T_s = p_s_end - p_s_start
        if T_s <= 0:
            continue

        x_period = x_full[p_s_start:p_s_end]
        valid_period = valid_full[p_s_start:p_s_end]
        sampled_frame_ids = sampled_frame_ids_full[p_s_start:p_s_end]

        dead_logits_accum = np.zeros(T_s, dtype=np.float32)
        team_logits_accum = np.zeros((T_s, 2), dtype=np.float32)
        edge_logits_accum = np.zeros((T_s, 576), dtype=np.float32)
        weight_accum = np.zeros(T_s, dtype=np.float32)

        starts = list(range(0, max(1, T_s - window_len_samples + 1), stride_samples))
        if len(starts) == 0 or starts[-1] + window_len_samples < T_s:
            starts.append(max(0, T_s - window_len_samples))

        infer_batch_size = 32
        for i in range(0, len(starts), infer_batch_size):
            chunk_starts = starts[i : i + infer_batch_size]
            b_x_list = []
            b_v_list = []
            meta = []

            for st in chunk_starts:
                en = min(st + window_len_samples, T_s)
                w_len = en - st
                sub_x = x_period[st:en]
                sub_v = valid_period[st:en]

                if w_len < window_len_samples:
                    pad = window_len_samples - w_len
                    sub_x = torch.nn.functional.pad(sub_x, (0, 0, 0, 0, 0, pad))
                    sub_v = torch.nn.functional.pad(sub_v, (0, 0, 0, pad))

                b_x_list.append(sub_x)
                b_v_list.append(sub_v)
                meta.append((st, en, w_len))

            b_x = torch.stack(b_x_list, dim=0)
            b_v = torch.stack(b_v_list, dim=0)

            with torch.no_grad():
                outputs = model(b_x, b_v)

            d_l_all = outputs.dead_logit.cpu().numpy()
            t_l_all = outputs.team_logits.cpu().numpy()
            e_l_all = outputs.compressed_edge_logits.cpu().numpy()

            for idx, (st, en, w_len) in enumerate(meta):
                w_weights = (
                    ramp_w_full[:w_len]
                    if w_len == window_len_samples
                    else _window_ramp_weights(w_len, overlap_samples if len(starts) > 1 else 0)
                )
                dead_logits_accum[st:en] += d_l_all[idx, :w_len] * w_weights
                team_logits_accum[st:en] += t_l_all[idx, :w_len] * w_weights[:, None]
                edge_logits_accum[st:en] += e_l_all[idx, :w_len] * w_weights[:, None]
                weight_accum[st:en] += w_weights

        np.maximum(weight_accum, np.float32(1e-5), out=weight_accum)
        period_results.append(
            PeriodLogits(
                p_start=p_start,
                p_end=p_end,
                n_p_frames=n_p_frames,
                sampled_frame_ids=sampled_frame_ids,
                dead_logits=torch.tensor(dead_logits_accum / weight_accum, device=device),
                team_logits=torch.tensor(team_logits_accum / weight_accum[:, None], device=device),
                edge_logits=torch.tensor(edge_logits_accum / weight_accum[:, None], device=device),
            )
        )

    return period_results


def fuse_and_upsample_n020_logits(
    period_logits: list[PeriodLogits],
    total_frames: int,
    sample_fps: float,
    model: PathCRFNet,
    path_weight: float = 0.5,
    device: torch.device = CPU_DEVICE,
) -> NDArray[Any]:
    """Fuse period logits under given path_weight and upsample to 25Hz."""
    comp_src = model.comp_src.to(device)
    comp_dst = model.comp_dst.to(device)
    probs_25hz = np.zeros((total_frames, 3), dtype=np.float32)
    probs_25hz[:] = 1.0 / 3.0

    for pl in period_logits:
        fused_probs = (
            fuse_n020_probabilities(
                pl.dead_logits,
                pl.team_logits,
                compressed_edge_logits=pl.edge_logits,
                comp_src=comp_src,
                comp_dst=comp_dst,
                path_weight=path_weight,
                team_size=model.team_size,
            )
            .cpu()
            .numpy()
        )

        T_s = len(fused_probs)
        if sample_fps >= 25.0 or T_s == pl.n_p_frames:
            probs_25hz[pl.p_start : pl.p_end] = fused_probs[: pl.n_p_frames]
        else:
            target_25hz_ids = np.arange(pl.p_start, pl.p_end, dtype=np.float64)
            src_ids = pl.sampled_frame_ids.astype(np.float64)
            up_probs = np.zeros((pl.n_p_frames, 3), dtype=np.float32)
            for c in range(3):
                up_probs[:, c] = np.interp(target_25hz_ids, src_ids, fused_probs[:, c])
            up_probs = up_probs / np.clip(up_probs.sum(axis=-1, keepdims=True), 1e-8, None)
            probs_25hz[pl.p_start : pl.p_end] = up_probs

    return probs_25hz


@torch.no_grad()  # type: ignore[untyped-decorator]
def predict_n020_probabilities(
    model: PathCRFNet,
    match: CanonicalMatch,
    *,
    sample_fps: float = 25.0,
    output_fps: float = 25.0,
    device: torch.device = CPU_DEVICE,
    window_seconds: float = 10.0,
    overlap_seconds: float = 2.0,
    path_weight: float = 0.5,
    precomputed_batch: PathCRFBatch | None = None,
) -> NDArray[Any]:
    """Compute calibrated frame-level probabilities [N_frames, 3] ordered [H, A, D].

    Never inspects match.labels or uses ground-truth ball state/episodes.
    """
    # The explicit model/state-machine structure keeps the domain cases auditable.
    # pylint: disable=too-many-arguments
    if output_fps != 25.0:
        raise ValueError("N020 output is aligned to the canonical 25 Hz timeline")
    period_logits = extract_n020_match_logits(
        model,
        match,
        sample_fps=sample_fps,
        device=device,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        precomputed_batch=precomputed_batch,
    )
    return fuse_and_upsample_n020_logits(
        period_logits,
        total_frames=match.n_frames,
        sample_fps=sample_fps,
        model=model,
        path_weight=path_weight,
        device=device,
    )


def decode_n020_match(
    probs_25hz: NDArray[Any],
    period: NDArray[Any],
    decoder: DecodeConfig | None = None,
) -> NDArray[Any]:
    """Apply Viterbi transition decoding to 25Hz probabilities.

    Returns:
        labels: [N_frames] string array with values in {'H', 'A', 'D', 'X'}.
    """
    if decoder is None:
        decoder = DecodeConfig(team_switch=2.0, dead_switch=4.0, transition_gain=0.0)

    log_probs = np.log(np.clip(probs_25hz, 1e-7, 1.0)).astype(np.float32)
    int_labels = decode_match(log_probs, period.astype(np.int8), decoder)

    label_chars = np.array(["H", "A", "D", "X"], dtype="<U1")
    return label_chars[int_labels]
