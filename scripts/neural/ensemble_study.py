"""Probability-averaging ensemble of runs with cached log-probs (same frozen splits required).

Reads ``artifacts/neural/runs/<run>/logp_cache.npz`` written by ``decoder_study.py`` for each run,
averages per-frame probabilities, selects the symmetric decoder on external validation, and
scores the external test split once. The challenge ground truth is never read.

Usage::

    python scripts/neural/ensemble_study.py --runs n002_set_temporal_mixed n003_set_temporal_large
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "neural"))

import numpy as np  # noqa: E402
from decoder_study import DEAD, METRICS, TEAM, decode_all, sym  # noqa: E402

from possession_lab.neural.canonical import CanonicalMatch  # noqa: E402
from possession_lab.neural.external_eval import official_metrics, score  # noqa: E402


def main() -> int:
    """Run the ensemble study and write ``artifacts/neural/ensembles/<name>.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    runs_dir = ROOT / "artifacts" / "neural" / "runs"
    caches = [dict(np.load(runs_dir / r / "logp_cache.npz")) for r in args.runs]
    with (runs_dir / args.runs[0] / "config.toml").open("rb") as handle:
        data_cfg = tomllib.load(handle)["data"]
    for r in args.runs[1:]:
        with (runs_dir / r / "config.toml").open("rb") as handle:
            other = tomllib.load(handle)["data"]
        if (other["val"], other["test"]) != (data_cfg["val"], data_cfg["test"]):
            raise ValueError(f"{r} uses different splits; ensemble not comparable")

    keys = sorted(set.intersection(*(set(c) for c in caches)))
    logp = {
        k: (
            np.logaddexp.reduce(np.stack([c[k] for c in caches]), axis=0) - np.log(len(caches))
        ).astype(np.float32)
        for k in keys
    }
    canon = ROOT / data_cfg["canonical_dir"]
    labels = {
        n: CanonicalMatch.load(canon / f"{n}.npz") for n in data_cfg["val"] + data_cfg["test"]
    }

    params = list(itertools.product(TEAM, DEAD))
    costs = np.stack([sym(*p) for p in params])
    pm = official_metrics()
    val_scores = []
    for name in data_cfg["val"]:
        m = labels[name]
        runs = pm.valid_runs(m.labels)
        for cond in ("clean", "degraded"):
            paths = decode_all(logp[f"{name}|{cond}"], m.period, costs)
            val_scores.append(
                [
                    pm.spell_scores(m.labels, pm.fill_abstentions(p, runs), runs)["mean_f1"]
                    for p in paths
                ]
            )
    val_means = np.asarray(val_scores).mean(axis=0)
    best = int(np.argmax(val_means))
    per: dict[str, dict[str, float]] = {}
    for name in data_cfg["test"]:
        m = labels[name]
        for cond in ("clean", "degraded"):
            path = decode_all(logp[f"{name}|{cond}"], m.period, costs[best : best + 1])[0]
            r = score(m.labels, path)
            per[f"{name}|{cond}"] = {k: float(r[k]) for k in METRICS}
    groups = {
        "all": list(per),
        "skillcorner_clean": [
            k for k in per if k.startswith("skillcorner") and k.endswith("clean")
        ],
        "idsse_degraded": [k for k in per if k.startswith("idsse") and k.endswith("degraded")],
    }
    test = {
        g: {m: float(np.mean([per[k][m] for k in ks])) for m in METRICS}
        for g, ks in groups.items()
        if ks
    }
    line = f"ensemble {'+'.join(args.runs)} params={params[best]} val={val_means[best]:.4f}"
    for g, t in test.items():
        line += f" | {g}: spellF1={t['spell_f1_mean']:.4f} macroF1={t['macro_f1']:.4f} n={t['n_pred_spells']:.0f}/{t['n_gt_spells']:.0f}"
    print(line)
    out = ROOT / "artifacts" / "neural" / "ensembles" / f"{'+'.join(args.runs)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "runs": args.runs,
                "params": params[best],
                "val_spell_f1": float(val_means[best]),
                "test": test,
                "per_item": per,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
