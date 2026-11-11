"""Train a set-temporal possession model on external canonical matches.

Usage::

    .venv-neural/Scripts/python scripts/neural/train.py --config configs/neural/n001_set_temporal_base.toml
        [--steps N] [--resume] [--run-name NAME]

Writes ``artifacts/neural/runs/<run>/{last.pt,best.pt,metrics.jsonl,config.toml}`` (gitignored).
Validation uses the external validation split only; the challenge ground truth is never read.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from possession_lab.neural.canonical import CanonicalMatch  # noqa: E402
from possession_lab.neural.data import WindowConfig, WindowStream, degrade_match, predict_log_probs  # noqa: E402
from possession_lab.neural.decode import DecodeConfig, decode_match  # noqa: E402
from possession_lab.neural.external_eval import score  # noqa: E402
from possession_lab.neural.features import feature_dims  # noqa: E402
from possession_lab.neural.model import ModelConfig, SetTemporalNet, count_parameters  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    """Read the TOML experiment config."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def boundary_weights(labels: torch.Tensor, weight: float, radius: int) -> torch.Tensor:
    """Per-frame loss weights ``[B, T]``: ``weight`` within ``radius`` frames of a label change.

    Changes into or out of ``X`` (gap/unlabelled) are ignored.
    """
    change = (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] != 3) & (labels[:, :-1] != 3)
    marks = torch.zeros_like(labels, dtype=torch.float32)
    marks[:, 1:] = change.float()
    near = torch.nn.functional.max_pool1d(
        marks.unsqueeze(1), kernel_size=2 * radius + 1, stride=1, padding=radius
    ).squeeze(1)
    return 1.0 + (weight - 1.0) * near


def transition_targets(labels: torch.Tensor, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Possession-change targets ``[B, T]`` and their validity mask (N012 transition head).

    Target 1 within ``radius`` frames of a *direct* H<->A label change (a turnover between teams,
    taken from the external ground truth); 0 elsewhere. Frames labelled ``X`` (gap / unlabelled /
    masked low-detection) are excluded, so changes into or out of ``X`` never create targets.
    """
    team = labels <= 1
    change = team[:, 1:] & team[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    marks = torch.zeros_like(labels, dtype=torch.float32)
    marks[:, 1:] = change.float()
    target = torch.nn.functional.max_pool1d(
        marks.unsqueeze(1), kernel_size=2 * radius + 1, stride=1, padding=radius
    ).squeeze(1)
    return target, (labels != 3).float()


def latent_ball_loss(
    out: torch.Tensor, target: torch.Tensor, visible_weight: float
) -> torch.Tensor:
    """Gaussian NLL of the true ball position (N014 latent-ball head).

    ``out [B, T, 4]`` = predicted mean x, y (scaled like the ball token) and log-variances;
    ``target [B, T, 4]`` = true x, y (scaled), target valid, input ball hidden. Frames where the
    input ball is hidden weigh 1, frames where it is visible ``visible_weight``; frames without a
    true ball (the source itself lost it) are ignored.
    """
    mean, logvar = out[..., :2].float(), out[..., 2:4].float().clamp(-8.0, 6.0)
    nll = 0.5 * (logvar + (target[..., :2] - mean) ** 2 / logvar.exp()).sum(-1)
    hidden = target[..., 3]
    weight = target[..., 2] * (hidden + visible_weight * (1.0 - hidden))
    return (nll * weight).sum() / weight.sum().clamp_min(1.0)


def latent_ball_huber(out: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    """Bounded latent-ball loss (D-019): Huber on the predicted mean, input-hidden frames only.

    N014's NLL is unbounded below where the ball is visible (copy it, shrink the variance), so it
    dominated training. Here the loss is >= 0, ignores the log-variance channels, and is computed
    only where augmentation hid the input ball and the source ball is known.
    """
    weight = target[..., 2] * target[..., 3]
    err = torch.nn.functional.smooth_l1_loss(
        out[..., :2].float(), target[..., :2], beta=beta, reduction="none"
    ).sum(-1)
    return (err * weight).sum() / weight.sum().clamp_min(1.0)


def evaluate_val(
    model: torch.nn.Module, matches: list[CanonicalMatch], cfg: dict[str, Any], device: torch.device
) -> dict[str, float]:
    """External-validation spellF1 / macroF1 with the configured default decoder."""
    decode_cfg = DecodeConfig(**cfg["decode"])
    out: dict[str, list[float]] = {"spell_f1_mean": [], "macro_f1": [], "frame_acc": []}
    for match in matches:
        logp = predict_log_probs(
            model,
            match,
            cfg["train"]["window"],
            cfg["train"]["eval_stride"],
            device,
            cfg["train"]["max_players"],
        )
        pred = decode_match(logp, match.period, decode_cfg)
        result = score(match.labels, pred)
        graded = match.labels != 3
        out["spell_f1_mean"].append(result["spell_f1_mean"])
        out["macro_f1"].append(result["macro_f1"])
        out["frame_acc"].append(
            float((np.argmax(logp[:, :3], 1)[graded] == match.labels[graded]).mean())
        )
    return {k: float(np.mean(v)) for k, v in out.items()}


def main() -> int:  # noqa: PLR0915 - linear training script
    """Run (or resume) one training job."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    steps = args.steps or tcfg["steps"]
    run = args.run_name or cfg["experiment"]["name"]
    run_dir = ROOT / "artifacts" / "neural" / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.toml")

    seed = int(cfg["experiment"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    canon = ROOT / cfg["data"]["canonical_dir"]
    train_paths = [canon / f"{name}.npz" for name in cfg["data"]["train"]]
    val_matches = [
        degrade_match(CanonicalMatch.load(canon / f"{name}.npz"), cfg["data"]["degrade_eval_seed"])
        for name in cfg["data"]["val"]
    ]

    feature_version = int(tcfg.get("feature_version", 1))
    model_kwargs = dict(cfg["model"])
    if feature_version >= 2:
        player_dim, context_dim = feature_dims(feature_version)
        model_kwargs.update(player_features=player_dim, context_features=context_dim)
    model_cfg = ModelConfig(**model_kwargs)
    model = SetTemporalNet(model_cfg).to(device)
    print(f"device={device} params={count_parameters(model):,}", flush=True)
    optim = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    def lr_at(step: int) -> float:
        """Compute the warmup and cosine-decay learning rate for one training step."""
        if step < tcfg["warmup"]:
            return (step + 1) / tcfg["warmup"]
        progress = (step - tcfg["warmup"]) / max(1, steps - tcfg["warmup"])
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    start_step, best = 0, -1.0
    if args.resume and (run_dir / "last.pt").is_file():
        state = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_step, best = state["step"], state["best"]
        print(f"resumed at step {start_step} (best {best:.4f})", flush=True)
    elif tcfg.get("init_from"):
        # Fine-tuning (D-023): start from another run's weights with a fresh optimiser and
        # schedule. The architecture must match exactly (strict load).
        parent = ROOT / tcfg["init_from"]
        state = torch.load(parent, map_location=device, weights_only=False)
        if ModelConfig(**state["model_cfg"]) != model_cfg:
            raise ValueError(f"init_from {parent}: model config differs")
        model.load_state_dict(state["model"])
        print(f"initialised from {parent} (parent step {state.get('step')})", flush=True)

    stream = WindowStream(
        train_paths,
        WindowConfig(
            window=tcfg["window"],
            batch_size=tcfg["batch_size"],
            seed=seed + start_step,
            max_players=tcfg["max_players"],
            clean_prob=tcfg["clean_prob"],
            feature_version=feature_version,
            min_players_label=int(tcfg.get("min_players_label", 0)),
            ball_target=model_cfg.ball_head,
        ),
    )
    loader = DataLoader(
        stream,
        batch_size=None,
        num_workers=tcfg["num_workers"],
        persistent_workers=tcfg["num_workers"] > 0,
        pin_memory=True,
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=3)
    frame_loss = torch.nn.CrossEntropyLoss(ignore_index=3, reduction="none")
    boundary_weight = float(tcfg.get("boundary_weight", 1.0))
    boundary_radius = int(tcfg.get("boundary_radius", 12))
    transition_weight = float(tcfg.get("transition_weight", 1.0))
    transition_radius = int(tcfg.get("transition_radius", 2))
    ball_weight = float(tcfg.get("ball_weight", 0.5))
    ball_visible_weight = float(tcfg.get("ball_visible_weight", 0.2))
    ball_loss = str(tcfg.get("ball_loss", "nll"))  # "nll" (N014) or "huber" (D-019)
    ball_beta = float(tcfg.get("ball_huber_beta", 0.05))
    if ball_loss not in ("nll", "huber"):
        raise ValueError(f"unknown ball_loss {ball_loss!r}")
    ball_channel = 3 + int(model_cfg.transition_head)
    metrics_file = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    model.train()
    tick, running = time.perf_counter(), []
    accum = int(tcfg.get("accum", 1))
    batches = iter(loader)
    for step, first in enumerate(batches, start=start_step):
        if step >= steps:
            break
        optim.zero_grad(set_to_none=True)
        micro = [first] + [next(batches) for _ in range(accum - 1)]
        for batch in micro:
            ball, players, mask, labels = batch[:4]
            context = batch[4].to(device, non_blocking=True) if len(batch) > 4 else None
            ball_tgt = batch[5].to(device, non_blocking=True) if len(batch) > 5 else None
            ball, players = (
                ball.to(device, non_blocking=True),
                players.to(device, non_blocking=True),
            )
            mask, labels = mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(ball, players, mask, context)
                aux = None
                if model_cfg.ball_head and ball_tgt is not None:
                    ball_out = logits[..., ball_channel : ball_channel + 4]
                    aux = ball_weight * (
                        latent_ball_huber(ball_out, ball_tgt, ball_beta)
                        if ball_loss == "huber"
                        else latent_ball_loss(ball_out, ball_tgt, ball_visible_weight)
                    )
                if model_cfg.transition_head:
                    target, valid_t = transition_targets(labels, transition_radius)
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits[..., 3].float(), target, reduction="none"
                    )
                    tr_loss = transition_weight * (bce * valid_t).sum() / valid_t.sum().clamp_min(1)
                    aux = tr_loss if aux is None else aux + tr_loss
                logits = logits[..., :3]
                if boundary_weight == 1.0:
                    loss = loss_fn(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
                else:
                    per_frame = frame_loss(logits.float().permute(0, 2, 1), labels)
                    weights = boundary_weights(labels, boundary_weight, boundary_radius)
                    valid = (labels != 3).float()
                    loss = (per_frame * weights * valid).sum() / (weights * valid).sum().clamp_min(
                        1
                    )
                if aux is not None:
                    loss = loss + aux
            scaler.scale(loss / accum).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        scaler.step(optim)
        scaler.update()
        sched.step()
        running.append(float(loss))
        if (step + 1) % 100 == 0:
            rate = 100 / (time.perf_counter() - tick)
            mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
            print(
                f"step {step + 1} loss {np.mean(running):.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"{rate:.1f} it/s peak_mem {mem:.0f}MiB",
                flush=True,
            )
            metrics_file.write(
                json.dumps({"step": step + 1, "loss": float(np.mean(running))}) + "\n"
            )
            tick, running = time.perf_counter(), []
        if (step + 1) % tcfg["eval_every"] == 0 or step + 1 == steps:
            val = evaluate_val(model, val_matches, cfg, device)
            model.train()
            improved = val["spell_f1_mean"] > best
            best = max(best, val["spell_f1_mean"])
            print(f"  val@{step + 1} {json.dumps(val)} {'*' if improved else ''}", flush=True)
            metrics_file.write(json.dumps({"step": step + 1, "val": val}) + "\n")
            metrics_file.flush()
            state = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step + 1,
                "best": best,
                "model_cfg": model_cfg.to_dict(),
                "config": cfg,
            }
            torch.save(state, run_dir / "last.pt")
            if improved:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_cfg": model_cfg.to_dict(),
                        "config": cfg,
                        "step": step + 1,
                        "val": val,
                    },
                    run_dir / "best.pt",
                )
    metrics_file.close()
    print(f"done; best external-val spellF1 {best:.4f}; run dir {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
