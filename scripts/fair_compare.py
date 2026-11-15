"""Score the HMM, a neural checkpoint and smooth50 on the same degraded open matches.

Uses DFL matches J03WN1 and J03WR9, which no model trained or validated on, degraded the way
open_data_check.py does it: the ball is lost in long gaps, mostly during stoppages, like in our
tracking. Scores come from the unchanged official metric code.

    python scripts/fair_compare.py --ckpt artifacts/neural/runs/n013_n012_seed2/best.pt \
        --decoder 2 8 4 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "evaluation")]

import open_data_check as odc  # noqa: E402
import possession_metrics as pm  # noqa: E402

from possession_lab.atd import Tracking  # noqa: E402
from possession_lab.hmm import HmmParams, decode, features  # noqa: E402
from possession_lab.neural.canonical import CanonicalMatch  # noqa: E402
from possession_lab.neural.data import predict_log_probs  # noqa: E402
from possession_lab.neural.decode import DecodeConfig, decode_match  # noqa: E402
from possession_lab.neural.model import ModelConfig, SetTemporalNet  # noqa: E402

QUANTUM = 1e-3  # same rounding as predict_challenge.py


def to_canonical(tr: Tracking, truth: np.ndarray) -> CanonicalMatch:
    """Convert degraded tracking the same way neural/atd.py reads our match."""
    length, width = odc.PITCH

    def metres(xy: np.ndarray) -> np.ndarray:
        out = np.empty_like(xy)
        out[..., 0] = (xy[..., 0] - 0.5) * length
        out[..., 1] = (0.5 - xy[..., 1]) * width
        return out

    ball = np.full((len(truth), 3), np.nan, np.float32)
    ball[:, :2] = metres(tr.ball[:, :2])
    players = metres(tr.players)
    return CanonicalMatch(
        match_id="fair_compare",
        provider="idsse",
        ball=ball,
        home=players[:, tr.slot_team == 0].astype(np.float32),
        away=players[:, tr.slot_team == 1].astype(np.float32),
        labels=truth.astype(np.int8),
        period=np.ones(len(truth), np.int8),
        pitch=(float(length), float(width)),
    )


def neural_codes(model: SetTemporalNet, train: dict, match: CanonicalMatch, decoder: DecodeConfig):
    """Inference and decoding as predict_challenge.py does them."""
    logp = predict_log_probs(
        model,
        match,
        train["window"],
        train["eval_stride"],
        torch.device("cpu"),
        train["max_players"],
    )
    logp = (np.round(logp / QUANTUM) * QUANTUM).astype(np.float32)
    return decode_match(logp, match.period, decoder)


def score(truth: list[np.ndarray], pred: list[np.ndarray]) -> tuple[float, float, int, int]:
    """spellF1, macroF1, predicted and true possessions over both halves of a match."""
    edges = np.cumsum([0] + [len(t) for t in truth])
    runs = list(zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True))
    gt, pr = np.concatenate(truth), np.concatenate(pred).astype(np.int8)
    spells = pm.spell_scores(gt, pr, runs)
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pr)))
    return spells["mean_f1"], macro, spells["n_pred"], spells["n_gt"]


def main() -> int:
    """Print per-match and mean scores for the three models."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--decoder",
        type=float,
        nargs=3,
        required=True,
        metavar=("TEAM", "DEAD", "GAIN"),
        help="the run's decoder from external_eval.json",
    )
    parser.add_argument("--matches", nargs="+", default=["J03WN1", "J03WR9"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = SetTemporalNet(ModelConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"])
    model.eval()
    decoder = DecodeConfig(*args.decoder)
    smooth = import_module("official_documentation.starter.baseline").smooth

    names = ("HMM", "neural", "smooth50")
    rows: dict[str, list[tuple[float, float, int, int]]] = {name: [] for name in names}
    for seed in args.seeds:
        for match in args.matches:
            rng = np.random.default_rng(seed)
            truth, preds = [], {name: [] for name in names}
            for half in odc.load_cached(match):
                tr = odc.degrade(half, rng, 1.0, 0.003)
                f = features(tr, odc.PITCH, odc.FPS)
                n = len(half["truth"])
                truth.append(half["truth"])
                preds["HMM"].append(pm.label_codes(decode(f, HmmParams(), [(0, n)], odc.FPS)))
                canon = to_canonical(tr, half["truth"])
                preds["neural"].append(
                    neural_codes(model, state["config"]["train"], canon, decoder)
                )
                nearest = np.where(f.visible, np.where(f.d_home <= f.d_away, 0, 1), -1)
                preds["smooth50"].append(pm.label_codes(smooth(nearest.astype(np.int8), 50)))
            for name in names:
                rows[name].append(score(truth, preds[name]))
                spell, macro, n_pred, n_gt = rows[name][-1]
                print(
                    f"seed {seed} {match} {name:<9} spellF1 {spell:.3f}  macroF1 {macro:.3f}  "
                    f"possessions {n_pred} (true {n_gt})",
                    flush=True,
                )
    for name in names:
        mean = np.mean(rows[name], axis=0)
        print(
            f"mean {name:<9} spellF1 {mean[0]:.3f}  macroF1 {mean[1]:.3f}  "
            f"possessions {mean[2]:.0f} (true {mean[3]:.0f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
