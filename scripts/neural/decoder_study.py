"""Decoder study on cached external log-probabilities (selection on external val only).

Caches per-frame log-probabilities of one checkpoint for the external val and test splits
(clean + degraded) to ``artifacts/neural/runs/<run>/logp_cache.npz``, then compares decoder
families. Each family's parameters are chosen on validation; the test split is scored once per
family. The challenge ground truth is never read.

Families:

- ``sym``: production 2-parameter Viterbi (team switch, dead switch);
- ``asym``: 3 parameters — team switch, entering dead ball, leaving dead ball.

(Temperature scaling is not a separate family: dividing log-probs by tau and renormalising per
frame leaves the Viterbi path unchanged except for scaling every cost by tau, already covered.)

All candidates are decoded in a single batched Viterbi pass per match.

Usage::

    python scripts/neural/decoder_study.py --run n003_set_temporal_large
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
sys.path.insert(0, str(ROOT / "scripts" / "neural"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from evaluate import load_model  # noqa: E402

from possession_lab.neural.canonical import CanonicalMatch, X  # noqa: E402
from possession_lab.neural.data import degrade_match, predict_log_probs  # noqa: E402
from possession_lab.neural.external_eval import official_metrics, score  # noqa: E402

TEAM = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
DEAD = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
METRICS = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells")


def viterbi_batch(log_probs: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Viterbi paths ``[C, T]`` for ``C`` cost matrices ``[C, 3, 3]`` (from, to) in one pass."""
    n, c = len(log_probs), len(costs)
    back = np.empty((n, c, 3), np.int8)
    score_ = np.repeat(log_probs[:1].astype(np.float64), c, axis=0)
    lp = log_probs.astype(np.float64)
    cols = np.arange(3)
    rows = np.arange(c)[:, None]
    for t in range(1, n):
        cand = score_[:, :, None] - costs
        arg = np.argmax(cand, axis=1)
        back[t] = arg
        score_ = cand[rows, arg, cols] + lp[t]
    paths = np.empty((c, n), np.int8)
    paths[:, -1] = np.argmax(score_, axis=1)
    for t in range(n - 1, 0, -1):
        paths[:, t - 1] = back[t, np.arange(c), paths[:, t]]
    return paths


def decode_all(logp: np.ndarray, period: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Per-period batched decoding; gap frames are X. Returns ``[C, T]``."""
    out = np.full((len(costs), len(period)), X, np.int8)
    for p in np.unique(period):
        if p == 0:
            continue
        idx = np.flatnonzero(period == p)
        out[:, idx] = viterbi_batch(logp[idx], costs)
    return out


def sym(team: float, dead: float) -> np.ndarray:
    """Symmetric cost matrix (production decoder)."""
    return np.array([[0, team, dead], [team, 0, dead], [dead, dead, 0]], float)


def asym(team: float, into_dead: float, out_dead: float) -> np.ndarray:
    """Asymmetric dead-ball entry (row H/A -> D) and exit (row D -> H/A) costs."""
    return np.array([[0, team, into_dead], [team, 0, into_dead], [out_dead, out_dead, 0]], float)


def main() -> int:
    """Cache log-probs if needed, then compare decoder families."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--ckpt", default="best.pt")
    args = parser.parse_args()
    run_dir = ROOT / "artifacts" / "neural" / "runs" / args.run
    cache = run_dir / "logp_cache.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir / args.ckpt, device)
    canon = ROOT / cfg["data"]["canonical_dir"]
    seed = cfg["data"]["degrade_eval_seed"]

    splits: dict[str, list[tuple[str, CanonicalMatch]]] = {"val": [], "test": []}
    for split, items in splits.items():
        for name in cfg["data"][split]:
            clean = CanonicalMatch.load(canon / f"{name}.npz")
            items.append((f"{name}|clean", clean))
            items.append((f"{name}|degraded", degrade_match(clean, seed)))

    stored = dict(np.load(cache)) if cache.is_file() else {}
    for split, items in splits.items():
        for key, match in items:
            if key not in stored:
                print(f"inference {split} {key}", flush=True)
                stored[key] = predict_log_probs(
                    model,
                    match,
                    cfg["train"]["window"],
                    cfg["train"]["eval_stride"],
                    device,
                    cfg["train"]["max_players"],
                )
    np.savez_compressed(cache, **stored)

    families = {
        "sym": [(p, sym(*p)) for p in itertools.product(TEAM, DEAD)],
        "asym": [(p, asym(*p)) for p in itertools.product(TEAM, DEAD, DEAD)],
    }
    report: dict[str, Any] = {}
    for fam, cands in families.items():
        costs = np.stack([c for _, c in cands])
        pm = official_metrics()
        val_scores = []
        for key, m in splits["val"]:
            runs = pm.valid_runs(m.labels)
            paths = decode_all(stored[key], m.period, costs)
            val_scores.append(
                [
                    pm.spell_scores(m.labels, pm.fill_abstentions(path, runs), runs)["mean_f1"]
                    for path in paths
                ]
            )
        val_means = np.asarray(val_scores).mean(axis=0)
        best = int(np.argmax(val_means))
        per_item: dict[str, list[dict[str, float]]] = {}
        for key, m in splits["test"]:
            path = decode_all(stored[key], m.period, costs[best : best + 1])[0]
            per_item[key] = [{k: float(score(m.labels, path)[k]) for k in METRICS}]
        best_i = 0
        test_keys = [k for k, _ in splits["test"]]
        groups = {
            "all": test_keys,
            "skillcorner_clean": [
                k for k in test_keys if k.startswith("skillcorner") and k.endswith("clean")
            ],
            "idsse_degraded": [
                k for k in test_keys if k.startswith("idsse") and k.endswith("degraded")
            ],
        }
        test = {
            g: {m: float(np.mean([per_item[k][best_i][m] for k in ks])) for m in METRICS}
            for g, ks in groups.items()
            if ks
        }
        report[fam] = {
            "params": cands[best][0],
            "val_spell_f1": float(val_means[best]),
            "test": test,
        }
        line = f"{fam:<5} params={cands[best][0]} val={val_means[best]:.4f}"
        for g, t in test.items():
            line += (
                f" | {g}: spellF1={t['spell_f1_mean']:.4f} macroF1={t['macro_f1']:.4f} "
                f"n={t['n_pred_spells']:.0f}/{t['n_gt_spells']:.0f}"
            )
        print(line, flush=True)
    (run_dir / "decoder_study.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
