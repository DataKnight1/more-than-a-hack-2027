"""Strict experimental TimesFM 2.5 feature generation and RigL dynamic sparsity.

Both are *real*, executed components. TimesFM weights are never silently replaced by
Hermite or linear interpolation. Missing packages/checkpoints raise immediately.
The TimesFM teacher runs in eval/no-grad on CPU; the student can train on CUDA.
This is a research pilot, NOT a claim of challenge-rule eligibility or speedups.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

TIMESFM_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"


class TimesFMRequiredError(RuntimeError):
    """Raised if the strict TimesFM dependency or weights cannot be loaded."""


class TimesFMBallFeatures:
    """Forecast contiguous missing observations from past-only 25Hz ball tokens.

    The 4 features are x, y ([-1, 1] pitch scale), confidence, and imputed flag.
    Observed coordinates are copied. For leading gaps with no history, the
    returned confidence is 0 (not an invented TimesFM prediction).
    """

    def __init__(
        self,
        *,
        context: int = 256,
        horizon: int = 128,
        confidence_tau_s: float = 1.5,
        fps: float = 25.0,
        per_core_batch_size: int = 128,
        backend: Any = None,
        device: Any = None,
    ) -> None:
        if context < 16 or horizon < 1 or confidence_tau_s <= 0 or fps <= 0:
            raise ValueError("Invalid TimesFM forecast configuration")
        self.context, self.horizon = context, horizon
        self.confidence_tau_s, self.fps = confidence_tau_s, fps
        self.forecast_calls = 0
        self.forecasted_frames = 0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if backend is not None:
            # Dependency injection is for deterministic offline unit tests only.
            self._model = backend
            return
        try:
            # Optional or cyclic dependency: load only on this path.
            # pylint: disable-next=import-outside-toplevel
            import timesfm  # type: ignore[import-not-found]

            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                TIMESFM_CHECKPOINT, torch_compile=False
            )
            model.compile(
                timesfm.ForecastConfig(
                    max_context=context,
                    max_horizon=horizon,
                    normalize_inputs=True,
                    per_core_batch_size=per_core_batch_size,
                    use_continuous_quantile_head=False,
                    force_flip_invariance=True,
                    infer_is_positive=False,  # pitch-centred x/y may be negative
                )
            )
            if hasattr(model, "model") and hasattr(model.model, "to"):
                model.model.to(self.device)
                model.model.eval()
            self._model = model
        except Exception as exc:
            raise TimesFMRequiredError(
                "TimesFM 2.5 is required; install timesfm[torch], download "
                f"{TIMESFM_CHECKPOINT} and verify network/cache access. "
                f"Original error: {exc}"
            ) from exc

    def _predict_batch(
        self, tasks: list[tuple[NDArray[Any], NDArray[Any], int]]
    ) -> list[NDArray[Any]]:
        """Invoke TimesFM weights for multiple gaps in parallel."""
        if not tasks:
            return []
        max_horizon = max(t[2] for t in tasks)
        if max_horizon < 1 or max_horizon > self.horizon:
            raise ValueError("Forecast chunk exceeds configured TimesFM horizon")
        expected_series = 2 * len(tasks)
        inputs = []
        for x, y, _ in tasks:
            inputs.append(x.astype(np.float32))
            inputs.append(y.astype(np.float32))
        try:
            result = self._model.forecast(horizon=max_horizon, inputs=inputs)
        except TypeError:
            result = self._model.forecast(max_horizon, inputs)
        point = np.asarray(result[0], dtype=np.float32)
        if (
            point.shape[0] != expected_series
            or point.shape[1] < max_horizon
            or not np.isfinite(point).all()
        ):
            raise RuntimeError(
                f"TimesFM returned invalid trajectory shape/values: {point.shape} "
                f"(expected {expected_series})"
            )
        self.forecast_calls += 1
        out = []
        for k, (_, _, n) in enumerate(tasks):
            self.forecasted_frames += n
            px = point[2 * k, :n]
            py = point[2 * k + 1, :n]
            out.append(np.column_stack([px, py]))
        return out

    def _predict(self, x: NDArray[Any], y: NDArray[Any], n: int) -> NDArray[Any]:
        """Invoke actual TimesFM weights for both trajectory axes."""
        return self._predict_batch([(x, y, n)])[0]

    def from_ball_tokens(self, token: NDArray[Any], batch_size: int = 64) -> NDArray[Any]:
        """Use first 9 original ball channels to generate 4 extra channels.

        Input shape [T,9+] with visibility in col0 and observed normalized xy
        in columns 1:3. No source GT or future observed values enter forecasts.
        """
        # The explicit model/state-machine structure keeps the domain cases auditable.
        # pylint: disable=too-many-branches
        token = np.asarray(token, dtype=np.float32)
        if token.ndim != 2 or token.shape[1] < 9:
            raise ValueError("TimesFM input must be [T,9+] ball tokens")
        t = len(token)
        seen = token[:, 0] > 0.5
        raw = token[:, 1:3]
        result = np.zeros((t, 4), dtype=np.float32)
        result[seen, :2] = raw[seen]
        result[seen, 2] = 1.0
        result[:, 3] = (~seen).astype(np.float32)
        if not t:
            return result
        # Sequential causal completion. Missing history is filled ONLY with
        # earlier TimesFM predictions, never with the following observed point.
        history = np.zeros((t, 2), np.float32)
        i = 0
        last_available = -1
        gaps: list[tuple[int, int, int]] = []
        while i < t:
            if seen[i]:
                history[i] = raw[i]
                last_available = i
                i += 1
                continue
            j = i
            while j < t and not seen[j]:
                j += 1
            if last_available >= 0:
                gaps.append((i, j, last_available))
            i = j
        if not gaps:
            return result

        for b_start in range(0, len(gaps), batch_size):
            chunk_gaps = gaps[b_start : b_start + batch_size]
            max_chunks = max(math.ceil((j - i) / self.horizon) for i, j, _ in chunk_gaps)
            for c_idx in range(max_chunks):
                tasks: list[tuple[NDArray[Any], NDArray[Any], int]] = []
                task_meta: list[tuple[int, int, int]] = []
                for i, j, la in chunk_gaps:
                    start = i + c_idx * self.horizon
                    if start >= j:
                        continue
                    end = min(j, start + self.horizon)
                    context = history[max(0, start - self.context) : start]
                    if len(context) < 16:
                        context = np.concatenate(
                            [np.repeat(context[:1], 16 - len(context), axis=0), context], axis=0
                        )
                    tasks.append((context[:, 0], context[:, 1], end - start))
                    task_meta.append((start, end, la))

                if not tasks:
                    continue

                preds = self._predict_batch(tasks)
                for (start, end, la), pred in zip(task_meta, preds, strict=True):
                    clipped = np.clip(pred, -1.0, 1.0)
                    history[start:end] = clipped
                    result[start:end, :2] = clipped
                    elapsed = (np.arange(start, end) - la) / self.fps
                    result[start:end, 2] = np.exp(-elapsed / self.confidence_tau_s)

        return result


_FORECASTER: TimesFMBallFeatures | None = None


def strict_timesfm_features(ball: torch.Tensor) -> torch.Tensor:
    """Generate TimesFM features for a [B,T,9 or 13] ball tensor.

    Deliberately on CPU with detached inputs: pretrained TimesFM is an offline
    teacher, not an end-to-end differentiable trainable component.
    """
    global _FORECASTER  # pylint: disable=global-statement
    if ball.ndim != 3 or ball.shape[-1] not in (9, 13):
        raise ValueError(f"Expected 9/13-channel ball tokens, got {tuple(ball.shape)}")
    if ball.shape[-1] == 13:
        return ball
    if _FORECASTER is None:
        _FORECASTER = TimesFMBallFeatures(device=ball.device)
    samples = ball[:, :, :9].detach().cpu().float().numpy()
    extra = np.stack([_FORECASTER.from_ball_tokens(s) for s in samples])
    # Always replace existing 4 channels with real TimesFM output, never blend
    # or silently fall back to Hermite interpolation.
    head = ball[:, :, :9]
    return torch.cat((head, torch.from_numpy(extra).to(ball.device, dtype=ball.dtype)), dim=-1)


def get_match_timesfm_13ch(match: Any, cache_dir: Path | None = None) -> NDArray[Any]:
    """Load or compute and disk-cache the complete 13-channel ball array for a match."""
    global _FORECASTER  # pylint: disable=global-statement

    if cache_dir is None:
        cache_dir = Path("artifacts/neural/cache/timesfm")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{match.match_id}_13ch.npy"
    if cache_file.is_file():
        arr = np.asarray(np.load(cache_file, allow_pickle=False))
        print(f"[TimesFM] loaded {match.match_id} from cache ({cache_file.name})", flush=True)
        return arr

    print(
        f"[TimesFM] computing 13-ch ball features for {match.match_id} "
        f"({len(match.ball)} frames)...",
        flush=True,
    )
    t0 = time.time()
    # Optional or cyclic dependency: load only on this path.
    # pylint: disable-next=import-outside-toplevel
    from possession_lab.neural.features import build_tokens

    base_ball, _, _ = build_tokens(
        match.ball,
        match.home,
        match.away,
        max_players=1,
        ball_forecast=False,
    )
    if _FORECASTER is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _FORECASTER = TimesFMBallFeatures(device=device)

    extra = _FORECASTER.from_ball_tokens(base_ball, batch_size=64)
    full_13ch = np.asarray(np.concatenate([base_ball[:, :9], extra], axis=-1), dtype=np.float32)
    np.save(cache_file, full_13ch)
    elapsed = time.time() - t0
    print(
        f"[TimesFM] computed {match.match_id} in {elapsed:.1f}s -> saved {cache_file.name}",
        flush=True,
    )
    return full_13ch


@dataclass(frozen=True)
class RigLSettings:
    """Parameters controlling sparsity and connection updates."""

    sparsity: float = 0.50
    update_interval: int = 100
    stop_step: int = 18000
    drop_fraction: float = 0.30
    seed: int = 20261119


class RigLSparsifier:
    """Magnitude-prune/gradient-regrow (RigL) on transformer Linear layers.

    This is a native PyTorch implementation of the RigL update rule, not the
    Google TensorFlow repository. Dense gradient computation is intentional:
    unconnected weights need gradients for regrowth. Sparse FLOP speedups are
    NOT guaranteed without sparse kernels.
    """

    def __init__(self, model: nn.Module, settings: RigLSettings):
        if not (0 < settings.sparsity < 1 and 0 < settings.drop_fraction <= 1):
            raise ValueError("RigL sparsity and drop_fraction must lie in (0,1)")
        if settings.update_interval < 1 or settings.stop_step < 1:
            raise ValueError("RigL interval and stop_step must be positive")
        self.settings = settings
        self.layers: list[tuple[str, nn.Linear]] = []
        gen = torch.Generator(device="cpu").manual_seed(settings.seed)
        for name, layer in model.named_modules():
            if not isinstance(layer, nn.Linear):
                continue
            if not (name.startswith("spatial.layers.") or name.startswith("temporal.layers.")):
                continue
            # The core PyTorch MultiheadAttention QKV is a packed Parameter,
            # not nn.Linear; only eligible FFN/out_proj layers are sparsified.
            if hasattr(layer, "rigl_mask"):
                mask = layer.rigl_mask
            else:
                mask = torch.rand(layer.weight.shape, generator=gen) >= settings.sparsity
                layer.register_buffer("rigl_mask", mask.to(layer.weight.device), persistent=True)
            self.layers.append((name, layer))
        if not self.layers:
            raise RuntimeError("RigL required, but no spatial/temporal Linear layers were found")
        self._inactive_grads: dict[str, torch.Tensor] = {}
        self.updates = 0
        self.apply_masks()

    def apply_masks(self) -> None:
        """Zero weights for connections that are currently inactive."""
        with torch.no_grad():
            for _, layer in self.layers:
                layer.weight.mul_(layer.rigl_mask)

    def before_optimizer_step(self) -> None:
        """Record dense gradients for growth, then zero inactive gradients."""
        self._inactive_grads.clear()
        for name, layer in self.layers:
            g = layer.weight.grad
            if g is None:
                continue
            mask = layer.rigl_mask
            self._inactive_grads[name] = g.detach().clone()
            g.mul_(mask)

    def after_optimizer_step(self, optim: torch.optim.Optimizer, step: int) -> None:
        """Maintain sparsity and update connections on the RigL schedule."""
        self.apply_masks()
        if step % self.settings.update_interval or step > self.settings.stop_step:
            return
        any_changed = False
        with torch.no_grad():
            for name, layer in self.layers:
                grad = self._inactive_grads.get(name)
                if grad is None:
                    continue
                weight, mask = layer.weight, layer.rigl_mask
                active = mask.flatten().nonzero().flatten()
                inactive = (~mask).flatten().nonzero().flatten()
                k = min(len(inactive), max(1, round(len(active) * self.settings.drop_fraction)))
                if k < 1:
                    continue
                magnitude = weight.detach().flatten().abs()[active]
                chosen_drop = active[torch.topk(magnitude, k, largest=False).indices]
                candidates = grad.abs().flatten()[inactive]
                chosen_grow = inactive[torch.topk(candidates, k, largest=True).indices]
                mask_flat = mask.flatten()
                mask_flat[chosen_drop] = False
                mask_flat[chosen_grow] = True
                weight.flatten()[chosen_drop] = 0.0
                weight.flatten()[chosen_grow] = 0.0
                # Reset AdamW moments on altered edges to avoid stale momentum.
                state = optim.state.get(weight, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state:
                        state[key].flatten()[chosen_drop] = 0
                        state[key].flatten()[chosen_grow] = 0
                any_changed = True
        if any_changed:
            self.updates += 1
        self.apply_masks()
        self._inactive_grads.clear()

    def stats(self) -> dict[str, float | int]:
        """Summarize update count and the current active connection ratio."""
        active = sum(int(layer.rigl_mask.sum()) for _, layer in self.layers)
        total = sum(layer.rigl_mask.numel() for _, layer in self.layers)
        return {
            "rigl_updates": self.updates,
            "rigl_active": active,
            "rigl_total": total,
            "rigl_sparsity_actual": 1 - active / total,
        }
