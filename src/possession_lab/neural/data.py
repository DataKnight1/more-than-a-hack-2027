"""Window sampling for training and deterministic match degradation for evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import IterableDataset, get_worker_info

from possession_lab.neural.augment import DegradeConfig, degrade, symmetry, symmetry_dir
from possession_lab.neural.canonical import CanonicalMatch, X
from possession_lab.neural.features import build_inputs, build_tokens, version_of
from possession_lab.neural.matchops import degrade_match, period_spans
from possession_lab.neural.team_features import attack_direction

__all__ = ["WindowConfig", "WindowStream", "degrade_match", "period_spans", "predict_log_probs"]

Batch = tuple[torch.Tensor, ...]  # (ball, players, mask, labels[, context][, ball_target])
BALL_SCALE = np.array([52.5, 34.0], np.float32)


@dataclass(frozen=True)
class WindowConfig:
    """Sampling and augmentation settings for :class:`WindowStream`."""

    window: int
    batch_size: int
    seed: int
    max_players: int = 30
    clean_prob: float = 0.1
    degrade_cfg: DegradeConfig = DegradeConfig()
    feature_version: int = 1
    ball_features: int = 9
    min_players_label: int = 0
    ball_target: bool = False

    def __post_init__(self) -> None:
        """Reject settings that would create empty or invalid batches."""
        if self.window < 1 or self.batch_size < 1 or self.max_players < 1:
            raise ValueError("window, batch_size and max_players must be positive")
        if self.seed < 0 or self.min_players_label < 0:
            raise ValueError("seed and min_players_label must be non-negative")
        if not 0.0 <= self.clean_prob <= 1.0:
            raise ValueError("clean_prob must be between 0 and 1")
        if self.ball_target and self.feature_version < 2:
            raise ValueError("ball_target requires feature_version >= 2")


class WindowStream(IterableDataset[Batch]):  # type: ignore[misc]
    """Infinite stream of augmented training batches from canonical matches.

    Each worker owns an independent ``numpy`` generator seeded from ``(seed, worker_id)``, so the
    stream is reproducible for a fixed worker count.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        config: WindowConfig,
    ) -> None:
        """Create a deterministic stream over canonical external match files.

        Args:
            paths: Canonical match files eligible for training.
            config: Window, batch, masking, and augmentation settings.
        """
        super().__init__()
        self.paths = list(paths)
        self.config = config

    def __iter__(self) -> Iterator[Batch]:
        """Yield indefinitely sampled and augmented training batches.

        Returns:
            An iterator of feature, label, mask, and optional auxiliary-target batches.

        Raises:
            ValueError: If no eligible match period is longer than the configured window.
        """
        info = get_worker_info()
        worker = info.id if info is not None else 0
        cfg = self.config
        rng = np.random.default_rng([cfg.seed, worker])
        matches = [CanonicalMatch.load(p) for p in self.paths]
        if cfg.min_players_label:
            matches = [mask_low_detection(m, cfg.min_players_label) for m in matches]
        directions = (
            {id(m): attack_direction(m.home, m.away, m.period) for m in matches}
            if cfg.feature_version >= 2
            else {}
        )
        spans = [
            (m, s) for m in matches for s in period_spans(m.period) if s[1] - s[0] > cfg.window
        ]
        if not spans:
            raise ValueError(
                f"no external match period is longer than the configured {cfg.window}-frame window"
            )
        weights = np.array([s[1] - s[0] for _, s in spans], dtype=np.float64)
        weights /= weights.sum()
        while True:
            balls: list[NDArray[np.float32]] = []
            players: list[NDArray[np.float32]] = []
            masks: list[NDArray[np.bool_]] = []
            labels: list[NDArray[np.int8]] = []
            contexts: list[NDArray[np.float32]] = []
            targets: list[NDArray[np.float32]] = []
            for _ in range(cfg.batch_size):
                match, (lo, hi) = spans[rng.choice(len(spans), p=weights)]
                start = int(rng.integers(lo, hi - cfg.window))
                sl = slice(start, start + cfg.window)
                strength = 0.0 if rng.random() < cfg.clean_prob else 1.0
                b, h, a = degrade(
                    rng, match.ball[sl], match.home[sl], match.away[sl], cfg.degrade_cfg, strength
                )
                ball_forecast = cfg.ball_features == 13 or cfg.feature_version == 5
                if cfg.feature_version >= 2:
                    clean_xy = match.ball[sl, :2] if cfg.ball_target else None
                    b, h, a, lab, hdir, *rest = symmetry_dir(
                        rng, b, h, a, match.labels[sl], directions[id(match)][sl], clean_xy
                    )
                    if cfg.ball_target:
                        # latent-ball target [T, 4]: true x, y (scaled), target valid, input hidden
                        txy = rest[0]
                        valid = ~np.isnan(txy[:, 0])
                        targets.append(
                            np.concatenate(
                                [
                                    np.nan_to_num(txy) / BALL_SCALE,
                                    valid[:, None],
                                    np.isnan(b[:, 0])[:, None],
                                ],
                                1,
                            ).astype(np.float32)
                        )
                    bf, pf, pm, ctx = build_inputs(
                        b,
                        h,
                        a,
                        cfg.max_players,
                        hdir,
                        cfg.feature_version,
                        ball_forecast=ball_forecast,
                    )
                    if ctx is None:
                        raise ValueError("feature version requires context features")
                    contexts.append(ctx)
                else:
                    b, h, a, lab = symmetry(rng, b, h, a, match.labels[sl])
                    bf, pf, pm = build_tokens(b, h, a, cfg.max_players, ball_forecast=ball_forecast)
                balls.append(bf)
                players.append(pf)
                masks.append(pm)
                labels.append(lab)
            batch: Batch = (
                torch.from_numpy(np.stack(balls)),
                torch.from_numpy(np.stack(players)),
                torch.from_numpy(np.stack(masks)),
                torch.from_numpy(np.stack(labels).astype(np.int64)),
            )
            if contexts:
                batch = (*batch, torch.from_numpy(np.stack(contexts)))
            if targets:
                batch = (*batch, torch.from_numpy(np.stack(targets)))
            yield batch


def mask_low_detection(match: CanonicalMatch, min_players: int) -> CanonicalMatch:
    """Copy ``match`` and set low-detection labels to ``X``.

    A frame is masked when fewer than ``min_players`` objects are tracked.

    EDA finding 7: broadcast replays / cut-aways (SkillCorner) show no players during dead ball, so
    such frames teach "nothing visible means dead ball", which is false for a fixed tactical
    camera. ``X`` frames are ignored by the loss. Observations are unchanged.
    """
    tracked = (~np.isnan(np.concatenate([match.home, match.away], axis=1)[..., 0])).sum(1)
    labels = match.labels.copy()
    labels[(tracked < min_players) & (match.period > 0)] = X
    return CanonicalMatch(
        match.match_id,
        match.provider,
        match.ball,
        match.home,
        match.away,
        labels,
        match.period,
        match.pitch,
        {**match.meta, "masked_low_detection": min_players},
    )


# PyTorch does not ship a PEP 561 marker, so the no-grad decorator is untyped to mypy.
@torch.no_grad()  # type: ignore[untyped-decorator]
def predict_log_probs(
    model: torch.nn.Module,
    match: CanonicalMatch,
    window: int,
    stride: int,
    device: torch.device,
    max_players: int = 30,
    batch_size: int = 16,
) -> NDArray[np.float32]:
    """Sliding-window inference with triangular blending; returns ``[T, 3]`` log-probabilities.

    With a transition head (N012+) a fourth column holds ``log p(possession change)`` per frame.
    Gap frames receive uniform probabilities (they are decoded to ``X`` anyway).
    """
    model.eval()
    version = version_of(getattr(getattr(model, "cfg", None), "context_features", 0))
    ball_features = getattr(getattr(model, "cfg", None), "ball_features", 9)
    ball_forecast = ball_features == 13
    home_dir = attack_direction(match.home, match.away, match.period) if version >= 2 else None
    n = match.n_frames
    cols = 4 if getattr(getattr(model, "cfg", None), "transition_head", False) else 3
    match_ball_13ch = None
    if getattr(getattr(model, "cfg", None), "timesfm_enabled", False):
        # Optional or cyclic dependency: load only on this path.
        # pylint: disable-next=import-outside-toplevel
        from possession_lab.neural.timesfm_rigl import get_match_timesfm_13ch

        match_ball_13ch = get_match_timesfm_13ch(match)
    acc = np.zeros((n, cols), np.float64)
    weight = np.zeros(n, np.float64)
    tri = np.minimum(np.arange(1, window + 1), np.arange(window, 0, -1)).astype(np.float64)
    for lo, hi in period_spans(match.period):
        starts = list(range(lo, max(lo + 1, hi - window + 1), stride))
        if starts[-1] + window < hi:
            starts.append(max(lo, hi - window))
        for i in range(0, len(starts), batch_size):
            group = starts[i : i + batch_size]
            feats = []
            for s in group:
                e = min(hi, s + window)
                inp = build_inputs(
                    match.ball[s:e],
                    match.home[s:e],
                    match.away[s:e],
                    max_players,
                    None if home_dir is None else home_dir[s:e],
                    version,
                    ball_forecast=(ball_forecast and match_ball_13ch is None),
                )
                if match_ball_13ch is not None:
                    inp = (match_ball_13ch[s:e], inp[1], inp[2], inp[3])
                feats.append(inp)
            length = min(len(f[0]) for f in feats)
            ball = torch.from_numpy(np.stack([f[0][:length] for f in feats])).to(device)
            players = torch.from_numpy(np.stack([f[1][:length] for f in feats])).to(device)
            mask = torch.from_numpy(np.stack([f[2][:length] for f in feats])).to(device)
            context_values = [f[3] for f in feats]
            context = None
            if version >= 2:
                if any(value is None for value in context_values):
                    raise ValueError("feature version requires context tensors for every window")
                context = torch.from_numpy(
                    np.stack([value[:length] for value in context_values if value is not None])
                ).to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(ball, players, mask, context)
            logits = logits.float()
            logp = torch.log_softmax(logits[..., :3], dim=-1)
            if cols == 4:
                logp = torch.cat([logp, torch.nn.functional.logsigmoid(logits[..., 3:4])], -1)
            logp = logp.cpu().numpy()
            for j, s in enumerate(group):
                w = tri[:length]
                acc[s : s + length] += logp[j] * w[:, None]
                weight[s : s + length] += w
    out = np.full((n, cols), np.log(1.0 / 3.0), np.float32)
    seen = weight > 0
    out[seen] = (acc[seen] / weight[seen, None]).astype(np.float32)
    return out


def graded_mask(labels: NDArray[np.int8]) -> NDArray[np.bool_]:
    """Frames that carry a supervised label."""
    return cast(NDArray[np.bool_], labels != X)
