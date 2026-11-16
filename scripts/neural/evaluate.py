"""External held-out evaluation: neural candidate vs organiser baseline on the same matches.

1. Decoder penalties are grid-searched on the *external validation* split (degraded condition).
2. The frozen external *test* split is scored once per condition (clean, degraded) for:
   neural + Viterbi, neural + argmax, baseline_raw and baseline_smooth50 (organiser semantics).

With several ``--run`` values the runs form a probability-averaging ensemble (D-022): per-frame
class and possession-change probabilities are averaged across models (latent-ball channels are
ignored) and the report is written to ``artifacts/neural/runs/ens_<run1>+<run2>...``.

The challenge ground truth is never read. Usage::

    .venv-neural/Scripts/python scripts/neural/evaluate.py --run n001_set_temporal_base
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from possession_lab.neural.baselines import baseline_codes  # noqa: E402
from possession_lab.neural.canonical import CanonicalMatch, X  # noqa: E402
from possession_lab.neural.data import degrade_match, predict_log_probs  # noqa: E402
from possession_lab.neural.decode import DecodeConfig, decode_match  # noqa: E402
from possession_lab.neural.external_eval import pooled, score  # noqa: E402
from possession_lab.neural.model import ModelConfig, SetTemporalNet  # noqa: E402

TEAM_GRID = (2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)
DEAD_GRID = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0)
# Transition-head runs only (stage 2, dead penalty fixed at its stage-1 value; D-016).
GAIN_GRID = (0.5, 1.0, 2.0, 4.0)


def load_model(ckpt: Path, device: torch.device) -> tuple[SetTemporalNet, dict[str, Any]]:
    """Restore a checkpoint written by ``train.py``."""
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = SetTemporalNet(ModelConfig(**state["model_cfg"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["config"]


def with_gaps_as_x(codes: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Force gap frames to X so predictions never span periods."""
    out = codes.copy()
    out[period == 0] = X
    return out


def average_log_probs(per_model: list[np.ndarray]) -> np.ndarray:
    """Probability average of per-frame log-probs across models (log-mean-exp per column).

    Uses the columns common to all models (3 classes, plus the possession-change column if every
    model has one). A single model is returned unchanged.
    """
    if len(per_model) == 1:
        return per_model[0]
    cols = min(lp.shape[1] for lp in per_model)
    stack = np.stack([lp[:, :cols] for lp in per_model])
    return (np.logaddexp.reduce(stack, axis=0) - np.log(len(per_model))).astype(np.float32)


def main() -> int:
    """Tune decoder on external val, then score the external test split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", nargs="+", required=True, help="one run, or several (ensemble)")
    parser.add_argument("--ckpt", default="best.pt")
    parser.add_argument(
        "--test",
        nargs="+",
        default=None,
        help="override test matches (e.g. unseen-provider transfer check)",
    )
    parser.add_argument("--tag", default="", help="suffix for the output JSON")
    args = parser.parse_args()
    runs_dir = ROOT / "artifacts" / "neural" / "runs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = [load_model(runs_dir / r / args.ckpt, device) for r in args.run]
    models = [m for m, _ in loaded]
    cfg = loaded[0][1]
    for r, (_, other) in zip(args.run[1:], loaded[1:], strict=True):
        same = all(other["data"][k] == cfg["data"][k] for k in ("val", "test", "degrade_eval_seed"))
        if not same or other["train"]["window"] != cfg["train"]["window"]:
            raise ValueError(f"{r}: splits / window differ from {args.run[0]}; not an ensemble")
    if len(args.run) == 1:
        run_dir = runs_dir / args.run[0]
    else:
        run_dir = runs_dir / ("ens_" + "+".join(args.run))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "members.json").write_text(json.dumps({"runs": args.run, "ckpt": args.ckpt}))
    tcfg, dcfg = cfg["train"], cfg["data"]
    canon = ROOT / dcfg["canonical_dir"]
    seed = dcfg["degrade_eval_seed"]

    def conditions(names: list[str]) -> dict[str, list[CanonicalMatch]]:
        """Load the requested clean and degraded external evaluation matches."""
        clean = [CanonicalMatch.load(canon / f"{n}.npz") for n in names]
        return {"clean": clean, "degraded": [degrade_match(m, seed) for m in clean]}

    def logps(matches: list[CanonicalMatch]) -> list[np.ndarray]:
        """Load cached per-match model log probabilities for the selected runs."""
        out = []
        for m in matches:
            per_model = [
                predict_log_probs(
                    model, m, tcfg["window"], tcfg["eval_stride"], device, tcfg["max_players"]
                )
                for model in models
            ]
            out.append(average_log_probs(per_model))
        return out

    # 1) decoder selection on external validation: mean spellF1 over clean + degraded conditions
    #    (rule fixed 2026-11-06 before any external-test result was produced)
    val_sets = conditions(dcfg["val"])
    val = val_sets["clean"] + val_sets["degraded"]
    val_lp = logps(val)

    def val_score(dec: DecodeConfig) -> float:
        """Read the external validation score used for decoder selection."""
        return float(
            np.mean(
                [
                    score(m.labels, decode_match(lp, m.period, dec))["spell_f1_mean"]
                    for m, lp in zip(val, val_lp, strict=True)
                ]
            )
        )

    grid = [
        (val_score(DecodeConfig(team, dead)), team, dead, 0.0)
        for team, dead in itertools.product(TEAM_GRID, DEAD_GRID)
    ]
    grid.sort(reverse=True)
    if val_lp and val_lp[0].shape[1] > 3:  # transition head: stage 2 over team x gain
        dead0 = grid[0][2]
        grid += [
            (val_score(DecodeConfig(team, dead0, gain)), team, dead0, gain)
            for team, gain in itertools.product(TEAM_GRID, GAIN_GRID)
        ]
        grid.sort(reverse=True)
    best_s, best_team, best_dead, best_gain = grid[0]
    decoder = DecodeConfig(best_team, best_dead, best_gain)
    print(
        f"decoder selected on external val: team={best_team} dead={best_dead} "
        f"gain={best_gain} (val spellF1 {best_s:.4f}; constant-penalty best "
        f"{max(g[0] for g in grid if g[3] == 0.0):.4f})",
        flush=True,
    )

    # 2) frozen external test
    report: dict[str, Any] = {
        "decoder": {
            "team_switch": best_team,
            "dead_switch": best_dead,
            "transition_gain": best_gain,
            "val_spell_f1": best_s,
            "val_spell_f1_constant_penalty": max(g[0] for g in grid if g[3] == 0.0),
        },
        "test": {},
    }
    test_names = args.test or dcfg["test"]
    for cond, matches in conditions(test_names).items():
        lps = logps(matches)
        systems: dict[str, dict[str, dict[str, Any]]] = {
            "neural_viterbi": {},
            "neural_argmax": {},
            "baseline_raw": {},
            "baseline_smooth50": {},
        }
        for m, lp in zip(matches, lps, strict=True):
            argmax = with_gaps_as_x(np.argmax(lp[:, :3], 1).astype(np.int8), m.period)
            systems["neural_viterbi"][m.match_id] = score(
                m.labels, decode_match(lp, m.period, decoder)
            )
            systems["neural_argmax"][m.match_id] = score(m.labels, argmax)
            for name, window in (("baseline_raw", 0), ("baseline_smooth50", 50)):
                codes = with_gaps_as_x(baseline_codes(m.ball, m.home, m.away, window), m.period)
                systems[name][m.match_id] = score(m.labels, codes)
        report["test"][cond] = {
            name: {"pooled": pooled(per), "per_match": per} for name, per in systems.items()
        }
        print(f"\n[{cond}] external test {test_names}")
        print(
            f"{'system':<20}{'spellF1':>9}{'macroF1':>9}{'n_pred':>8}{'n_gt':>7}{'alive/dead':>11}"
        )
        for name, entry in report["test"][cond].items():
            p = entry["pooled"]
            print(
                f"{name:<20}{p['spell_f1_mean']:>9.4f}{p['macro_f1']:>9.4f}"
                f"{p['n_pred_spells']:>8.0f}{p['n_gt_spells']:>7.0f}{p['alive_dead_f1']:>11.4f}"
            )
    (run_dir / f"external_eval{args.tag}.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    print(f"\nwritten {run_dir / f'external_eval{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
