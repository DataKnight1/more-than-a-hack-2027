"""Train N020 PathCRF model on external canonical matches.

Usage:
    .venv-neural/Scripts/python scripts/neural/train_n020_pathcrf.py --config configs/neural/n020_pathcrf_strongpaths.toml
        [--steps N] [--run-name NAME] [--resume] [--device cuda]

Features:
1. Strict player-only inputs (26 nodes: 22 players + 4 boundary nodes).
2. Data provenance firewall: strictly approved DFL train matches + SkillCorner + Metrica 1.
3. Strong path supervision on approved DFL matches; team/dead supervision across all matches.
4. AMP for Set Attention backbone, float32 for CRF sequence transitions.
5. In-memory pre-sampled feature caching for maximum GPU throughput.
6. Checkpoint saving with SHA-256 integrity verification.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.amp import GradScaler, autocast  # noqa: E402

from possession_lab.neural.canonical import CanonicalMatch, X  # noqa: E402
from possession_lab.neural.data import degrade_match  # noqa: E402
from possession_lab.neural.decode import DecodeConfig  # noqa: E402
from possession_lab.neural.external_eval import score  # noqa: E402
from possession_lab.neural.matchops import period_spans  # noqa: E402
from possession_lab.neural.pathcrf.adapter import PathCRFBatch, canonical_to_pathcrf  # noqa: E402
from possession_lab.neural.pathcrf.backbone import PathCRFNet  # noqa: E402
from possession_lab.neural.pathcrf.checkpoints import load_n020_checkpoint, save_n020_checkpoint  # noqa: E402
from possession_lab.neural.pathcrf.crf import DynamicSparseCRF  # noqa: E402
from possession_lab.neural.pathcrf.inference import decode_n020_match, predict_n020_probabilities  # noqa: E402
from possession_lab.neural.pathcrf.labels import generate_training_path_annotations  # noqa: E402
from possession_lab.neural.pathcrf.provenance import APPROVED_TRAIN_MATCHES, DENIED_MATCHES  # noqa: E402
from possession_lab.neural.pathcrf.supervision import LossWeights, compute_n020_loss  # noqa: E402


@dataclass
class CachedTrainMatch:
    match_id: str
    x: np.ndarray  # [T_s, 26, 8]
    valid: np.ndarray  # [T_s, 26]
    labels: np.ndarray  # [T_s]
    label_valid: np.ndarray  # [T_s]
    path_labels: np.ndarray  # [T_s]
    path_valid: np.ndarray  # [T_s]
    spans: list[tuple[int, int]]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def prepare_cached_matches(
    match_names: list[str],
    canonical_dir: Path,
    sample_fps: float,
    synced_parquet_dir: Path,
) -> list[CachedTrainMatch]:
    """Precompute and cache features, masks, and path labels in memory."""
    cached: list[CachedTrainMatch] = []
    print(
        f"Precomputing feature cache for {len(match_names)} training matches at {sample_fps} Hz...",
        flush=True,
    )

    for name in match_names:
        clean_id = name.replace("idsse_", "")
        if clean_id in DENIED_MATCHES or name in DENIED_MATCHES:
            raise ValueError(f"SECURITY VIOLATION: Denied match {name} in training list!")

        npz_path = canonical_dir / f"{name}.npz"
        if not npz_path.is_file():
            print(f"Warning: {npz_path} not found; skipping", flush=True)
            continue

        match = CanonicalMatch.load(npz_path)

        # 1. Strong path labels if approved
        path_ann = None
        if clean_id in APPROVED_TRAIN_MATCHES:
            path_ann = generate_training_path_annotations(
                clean_id, match.n_frames, synced_parquet_dir
            )

        path_edges = path_ann.edge_id if path_ann is not None else None
        path_conf = path_ann.confident if path_ann is not None else None

        # 2. Canonical to PathCRF batch
        batch = canonical_to_pathcrf(
            match,
            lo=0,
            hi=match.n_frames,
            target_fps=sample_fps,
            include_labels=True,
            path_annotations=path_edges,
            path_valid_mask=path_conf,
        )

        # 3. Find period spans in sampled domain
        sample_step = max(1, int(round(25.0 / sample_fps)))
        raw_spans = period_spans(match.period)
        s_spans = []
        for st, en in raw_spans:
            st_s = st // sample_step
            en_s = en // sample_step
            if en_s - st_s >= 20:  # minimum span
                s_spans.append((st_s, en_s))

        x_np = batch.x[0].numpy()
        v_np = batch.node_valid[0].numpy()
        lbl_np = batch.labels[0].numpy()
        lbl_v_np = batch.label_valid[0].numpy()

        p_lbl_np = (
            batch.path_labels[0].numpy()
            if batch.path_labels is not None
            else np.full(len(x_np), -1, dtype=np.int64)
        )
        p_v_np = (
            batch.path_valid[0].numpy()
            if batch.path_valid is not None
            else np.zeros(len(x_np), dtype=bool)
        )

        cached.append(
            CachedTrainMatch(
                match_id=name,
                x=x_np,
                valid=v_np,
                labels=lbl_np,
                label_valid=lbl_v_np,
                path_labels=p_lbl_np,
                path_valid=p_v_np,
                spans=s_spans,
            )
        )

    print(f"Feature cache ready: {len(cached)} matches cached.", flush=True)
    return cached


def sample_train_batch(
    cached_matches: list[CachedTrainMatch],
    batch_size: int,
    window: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a random batch of windows from cached matches."""
    b_x, b_v, b_l, b_lv, b_pl, b_pv = [], [], [], [], [], []

    for _ in range(batch_size):
        m_idx = rng.integers(0, len(cached_matches))
        m = cached_matches[m_idx]

        # Pick span
        sp_idx = rng.integers(0, len(m.spans))
        st_s, en_s = m.spans[sp_idx]

        max_start = en_s - window
        if max_start <= st_s:
            w_start = st_s
            w_end = min(en_s, st_s + window)
            pad_len = window - (w_end - w_start)
        else:
            w_start = rng.integers(st_s, max_start + 1)
            w_end = w_start + window
            pad_len = 0

        sub_x = m.x[w_start:w_end]
        sub_v = m.valid[w_start:w_end]
        sub_l = m.labels[w_start:w_end]
        sub_lv = m.label_valid[w_start:w_end]
        sub_pl = m.path_labels[w_start:w_end]
        sub_pv = m.path_valid[w_start:w_end]

        if pad_len > 0:
            sub_x = np.pad(sub_x, ((0, pad_len), (0, 0), (0, 0)))
            sub_v = np.pad(sub_v, ((0, pad_len), (0, 0)))
            sub_l = np.pad(sub_l, ((0, pad_len)), constant_values=X)
            sub_lv = np.pad(sub_lv, ((0, pad_len)), constant_values=False)
            sub_pl = np.pad(sub_pl, ((0, pad_len)), constant_values=-1)
            sub_pv = np.pad(sub_pv, ((0, pad_len)), constant_values=False)

        b_x.append(sub_x)
        b_v.append(sub_v)
        b_l.append(sub_l)
        b_lv.append(sub_lv)
        b_pl.append(sub_pl)
        b_pv.append(sub_pv)

    x_tensor = torch.tensor(np.stack(b_x), dtype=torch.float32, device=device)
    v_tensor = torch.tensor(np.stack(b_v), dtype=torch.bool, device=device)
    l_tensor = torch.tensor(np.stack(b_l), dtype=torch.long, device=device)
    lv_tensor = torch.tensor(np.stack(b_lv), dtype=torch.bool, device=device)
    pl_tensor = torch.tensor(np.stack(b_pl), dtype=torch.long, device=device)
    pv_tensor = torch.tensor(np.stack(b_pv), dtype=torch.bool, device=device)

    return x_tensor, v_tensor, l_tensor, lv_tensor, pl_tensor, pv_tensor


def evaluate_val_n020(
    model: PathCRFNet,
    val_data: list[tuple[CanonicalMatch, PathCRFBatch]],
    sample_fps: float,
    device: torch.device,
    decoder: DecodeConfig,
    path_weight: float = 0.5,
) -> dict[str, float]:
    """Score model on degraded external validation split."""
    model.eval()
    spell_f1s: list[float] = []
    macro_f1s: list[float] = []
    alive_dead_f1s: list[float] = []

    for m, pbatch in val_data:
        probs = predict_n020_probabilities(
            model,
            m,
            sample_fps=sample_fps,
            output_fps=25.0,
            device=device,
            window_seconds=10.0,
            overlap_seconds=2.0,
            path_weight=path_weight,
            precomputed_batch=pbatch,
        )
        labels_str = decode_n020_match(probs, m.period, decoder)
        char2code = {"H": 0, "A": 1, "D": 2, "X": 3}
        pred_codes = np.array([char2code[c] for c in labels_str], dtype=np.int8)

        res = score(m.labels, pred_codes)
        spell_f1s.append(res["spell_f1_mean"])
        macro_f1s.append(res["macro_f1"])
        alive_dead_f1s.append(res["alive_dead_f1"])

    return {
        "val_spell_f1": float(np.mean(spell_f1s)),
        "val_macro_f1": float(np.mean(macro_f1s)),
        "val_alive_dead_f1": float(np.mean(alive_dead_f1s)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment TOML config")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--run-name", default=None, help="Override run directory name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    mcfg = cfg["model"]
    dcfg = cfg["data"]

    steps = args.steps or tcfg["steps"]
    run_name = args.run_name or cfg["experiment"]["name"]
    run_dir = ROOT / "artifacts" / "neural" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.toml")

    seed = int(cfg["experiment"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    device = torch.device(args.device)

    print("================================================================================")
    print(f"N020 PATHCRF TRAINING: {run_name}")
    print(
        f"Device: {device} | Steps: {steps:,} | Window: {tcfg['window']} frames ({tcfg['sample_fps']} Hz)"
    )
    print("================================================================================")

    # 1. Model & CRF instantiation
    model = PathCRFNet(
        feat_dim=mcfg.get("feat_dim", 8),
        team_size=mcfg.get("team_size", 11),
        macro_dim=mcfg.get("macro_dim", 64),
        micro_dim=mcfg.get("micro_dim", 128),
        seq_dim=mcfg.get("seq_dim", 128),
        crf_edge_dim=mcfg.get("crf_edge_dim", 16),
        heads=mcfg.get("heads", 4),
        dropout=mcfg.get("dropout", 0.1),
    ).to(device)

    crf = DynamicSparseCRF(
        model.comp_src,
        model.comp_dst,
        edge_embed_dim=mcfg.get("crf_edge_dim", 16),
        team_size=mcfg.get("team_size", 11),
    ).to(device)

    # 2. Optimizer & Scheduler
    params = list(model.parameters()) + list(crf.parameters())
    optim = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    def lr_at(step: int) -> float:
        warmup = tcfg.get("warmup", 200)
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    # 3. Cache training matches
    canon_dir = ROOT / dcfg["canonical_dir"]
    synced_dir = ROOT / "artifacts" / "local" / "pathcrf_upstream" / "data" / "sportec_synced"
    cached_train = prepare_cached_matches(
        dcfg["train"], canon_dir, float(tcfg["sample_fps"]), synced_dir
    )

    # 4. Load validation matches (degraded condition)
    print("Loading external validation matches (degraded condition)...", flush=True)
    val_matches = [
        degrade_match(CanonicalMatch.load(canon_dir / f"{n}.npz"), dcfg["degrade_eval_seed"])
        for n in dcfg["val"]
    ]
    print("Precomputing validation feature cache...", flush=True)
    val_data = [
        (
            m,
            canonical_to_pathcrf(
                m,
                lo=0,
                hi=m.n_frames,
                target_fps=float(tcfg["sample_fps"]),
                include_labels=False,
            ),
        )
        for m in val_matches
    ]
    print(f"Validation cache ready: {len(val_data)} matches cached.", flush=True)

    loss_weights = LossWeights(
        w_dead=float(tcfg.get("w_dead", 1.0)),
        w_team=float(tcfg.get("w_team", 1.0)),
        w_edge=float(tcfg.get("w_edge", 0.5)),
        w_crf=float(tcfg.get("w_crf", 0.5)),
    )

    decoder = DecodeConfig(
        team_switch=float(cfg["decode"].get("team_switch", 2.0)),
        dead_switch=float(cfg["decode"].get("dead_switch", 4.0)),
        transition_gain=0.0,
    )
    path_weight = float(cfg["decode"].get("path_weight", 0.5))

    start_step = 0
    best_val_f1 = -1.0
    if args.resume and (run_dir / "last.pt").is_file():
        model, crf, state = load_n020_checkpoint(run_dir / "last.pt", device=device)
        start_step = state["step"]
        best_val_f1 = state.get("val_spell_f1", -1.0)
        print(
            f"Resumed from {run_dir / 'last.pt'} at step {start_step} (best val {best_val_f1:.4f})",
            flush=True,
        )

    rng = np.random.default_rng(seed + start_step)
    accum = int(tcfg.get("accum", 4))
    eval_every = int(tcfg.get("eval_every", 200))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    batch_size = int(tcfg["batch_size"])
    window = int(tcfg["window"])

    metrics_log = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    running_loss: list[float] = []
    t_start = time.perf_counter()

    print("\nStarting training loop...", flush=True)
    for step in range(start_step, steps):
        model.train()
        crf.train()

        # Gradient accumulation inner loop
        optim.zero_grad()
        accum_loss = 0.0

        for _ in range(accum):
            x, v, labels, lv, pl, pv = sample_train_batch(
                cached_train, batch_size, window, rng, device
            )

            with autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(x, v)
                loss, loss_dict = compute_n020_loss(
                    outputs,
                    labels,
                    lv,
                    path_labels=pl if loss_weights.w_edge > 0 or loss_weights.w_crf > 0 else None,
                    path_valid=pv if loss_weights.w_edge > 0 or loss_weights.w_crf > 0 else None,
                    crf=crf if loss_weights.w_crf > 0 else None,
                    weights=loss_weights,
                )
                scaled_loss = loss / accum

            scaler.scale(scaled_loss).backward()
            accum_loss += float(loss.item())

        scaler.unscale_(optim)
        nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(optim)
        scaler.update()
        sched.step()

        running_loss.append(accum_loss)

        # Logging
        if (step + 1) % 20 == 0 or step == start_step:
            lr_cur = sched.get_last_lr()[0]
            avg_loss = float(np.mean(running_loss[-20:]))
            fps = (20 * accum * batch_size * window) / max(0.1, time.perf_counter() - t_start)
            t_start = time.perf_counter()
            print(
                f"Step {step + 1:5d}/{steps} | Loss: {avg_loss:.4f} "
                f"(dead: {loss_dict.get('loss_dead', 0):.3f}, team: {loss_dict.get('loss_team', 0):.3f}, "
                f"edge: {loss_dict.get('loss_edge', 0):.3f}) | LR: {lr_cur:.2e} | {fps:.1f} frames/s",
                flush=True,
            )

        # Evaluation
        if (step + 1) % eval_every == 0 or (step + 1) == steps:
            print(f"\n--- Evaluating External Validation at Step {step + 1} ---", flush=True)
            val_metrics = evaluate_val_n020(
                model,
                val_data,
                sample_fps=float(tcfg["sample_fps"]),
                device=device,
                decoder=decoder,
                path_weight=path_weight,
            )
            val_spell = val_metrics["val_spell_f1"]
            val_macro = val_metrics["val_macro_f1"]
            val_ad = val_metrics["val_alive_dead_f1"]
            print(
                f"Validation SpellF1: {val_spell:.4f} | MacroF1: {val_macro:.4f} | Alive/Dead: {val_ad:.4f}",
                flush=True,
            )

            record = {
                "step": step + 1,
                "train_loss": float(np.mean(running_loss[-eval_every:])),
                **val_metrics,
                "timestamp": time.time(),
            }
            metrics_log.write(json.dumps(record) + "\n")
            metrics_log.flush()

            # Save last checkpoint
            save_n020_checkpoint(
                run_dir / "last.pt", model, crf, cfg, step=step + 1, val_spell_f1=val_spell
            )

            # Save best checkpoint
            if val_spell > best_val_f1:
                best_val_f1 = val_spell
                save_n020_checkpoint(
                    run_dir / "best.pt", model, crf, cfg, step=step + 1, val_spell_f1=val_spell
                )
                print(
                    f"--> NEW BEST MODEL: val_spell_f1 = {best_val_f1:.4f} saved to best.pt\n",
                    flush=True,
                )

    metrics_log.close()
    print("================================================================================")
    print(f"TRAINING COMPLETE: Best External Val SpellF1 = {best_val_f1:.4f}")
    print(f"Checkpoints written to: {run_dir}")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
