"""Exploratory analysis of possession behaviour on EXTERNAL training matches only.

Questions:

1. Which team behaviours separate "home has the ball" from "away has the ball"? For every team
   feature the in-possession and out-of-possession team values are compared and an AUC is
   computed for the home-minus-away difference (H vs A frames), overall and on frames where the
   ball is *not* observed (natural missingness, or degraded IDSSE).
2. Which features separate live from dead ball?
3. What happens around a possession change (event-locked profiles, -3 s .. +3 s)?

Only training-split matches are analysed, so feature design never sees validation/test matches,
and the challenge game is never read. Outputs: ``experiments/research/neural/EDA_POSSESSION.md`` (aggregate
tables only) and figures in ``artifacts/neural/eda/``.

Usage::

    python scripts/neural/eda_possession.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from possession_lab.neural.canonical import A, CanonicalMatch, D, H  # noqa: E402
from possession_lab.neural.matchops import degrade_match  # noqa: E402
from possession_lab.neural.team_features import (  # noqa: E402
    GLOBAL_FEATURES,
    N_TEAM,
    TEAM_FEATURES,
    attack_direction,
    context_features,
)

TRAIN = {
    "idsse": ["idsse_J03WMX", "idsse_J03WOH", "idsse_J03WOY", "idsse_J03WPY"],
    "skillcorner": [
        "skillcorner_1874553",
        "skillcorner_1899585",
        "skillcorner_1927964",
        "skillcorner_1953632",
        "skillcorner_1959846",
        "skillcorner_1996435",
        "skillcorner_2006229",
        "skillcorner_2007448",
        "skillcorner_2007721",
        "skillcorner_2010085",
        "skillcorner_2013725",
        "skillcorner_2016236",
    ],
    "metrica": ["metrica_1"],
}
STEP = 5  # analyse every 5th frame (0.2 s) to limit autocorrelation and memory
WINDOW = 75  # +/- 3 s around transitions


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC P(pos > neg) with ties counted half."""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def collect(names: list[str], degrade_ball: bool) -> dict[str, np.ndarray]:
    """Raw context features, labels, ball-seen flags and transition profiles for a provider."""
    feats, labels, seen = [], [], []
    trans: dict[str, list[np.ndarray]] = {"team_change": [], "dead_entry": [], "restart": []}
    canon = ROOT / "data" / "external" / "canonical"
    for name in names:
        match = CanonicalMatch.load(canon / f"{name}.npz")
        if degrade_ball:
            match = degrade_match(match, seed=11)
        direction = attack_direction(match.home, match.away, match.period)
        ctx = context_features(match.ball, match.home, match.away, direction, scaled=False)
        lab = match.labels
        ok = match.period > 0
        idx = np.flatnonzero(ok)[::STEP]
        feats.append(ctx[idx])
        labels.append(lab[idx])
        seen.append(~np.isnan(match.ball[idx, 0]))
        # event-locked profiles: features of the NEW possessor minus the OLD one (team changes)
        change = np.flatnonzero((lab[1:] != lab[:-1]) & ok[1:] & ok[:-1]) + 1
        for c in change:
            if c - WINDOW < 0 or c + WINDOW >= len(lab) or not ok[c - WINDOW : c + WINDOW].all():
                continue
            before, after = lab[c - 1], lab[c]
            seg = ctx[c - WINDOW : c + WINDOW]
            if before in (H, A) and after in (H, A):
                new, old = (
                    (seg[:, :N_TEAM], seg[:, N_TEAM : 2 * N_TEAM])
                    if after == H
                    else (seg[:, N_TEAM : 2 * N_TEAM], seg[:, :N_TEAM])
                )
                trans["team_change"].append(new - old)
            elif before in (H, A) and after == D:
                trans["dead_entry"].append(seg[:, 2 * N_TEAM :])
            elif before == D and after in (H, A):
                new, old = (
                    (seg[:, :N_TEAM], seg[:, N_TEAM : 2 * N_TEAM])
                    if after == H
                    else (seg[:, N_TEAM : 2 * N_TEAM], seg[:, :N_TEAM])
                )
                trans["restart"].append(np.concatenate([new - old, seg[:, 2 * N_TEAM :]], axis=1))
    out = {"x": np.concatenate(feats), "y": np.concatenate(labels), "seen": np.concatenate(seen)}
    for k, v in trans.items():
        out[k] = np.stack(v) if v else np.zeros((0, 2 * WINDOW, 1))
    return out


def team_table(data: dict[str, np.ndarray]) -> list[tuple[str, float, float, float, float]]:
    """(feature, median in-possession, median out-of-possession, AUC all, AUC ball-unseen)."""
    x, y, seen = data["x"], data["y"], data["seen"]
    rows = []
    for k, name in enumerate(TEAM_FEATURES):
        home, away = x[:, k], x[:, N_TEAM + k]
        att = np.concatenate([home[y == H], away[y == A]])
        dfn = np.concatenate([away[y == H], home[y == A]])
        diff = home - away
        a_all = auc(diff[y == H], diff[y == A])
        a_unseen = auc(diff[(y == H) & ~seen], diff[(y == A) & ~seen])
        rows.append((name, float(np.median(att)), float(np.median(dfn)), a_all, a_unseen))
    return rows


def dead_table(data: dict[str, np.ndarray]) -> list[tuple[str, float, float, float]]:
    """(feature, median alive, median dead, AUC alive-vs-dead) for global + summed team features."""
    x, y = data["x"], data["y"]
    alive, dead = np.isin(y, (H, A)), y == D
    cols = {name: x[:, 2 * N_TEAM + i] for i, name in enumerate(GLOBAL_FEATURES)}
    for k, name in enumerate(TEAM_FEATURES):
        if name in (
            "frac_high_intensity",
            "frac_sprint",
            "spread_length",
            "spread_width",
            "n_within_5m",
        ):
            cols[f"both_{name}"] = x[:, k] + x[:, N_TEAM + k]
    return [
        (n, float(np.median(v[alive])), float(np.median(v[dead])), auc(v[alive], v[dead]))
        for n, v in cols.items()
    ]


def plot_profiles(results: dict[str, dict[str, np.ndarray]], out_dir: Path) -> list[str]:
    """Event-locked mean profiles for selected features; returns written file names."""
    t = (np.arange(2 * WINDOW) - WINDOW) / 25.0
    written = []
    picks = [
        "closing_speed_near3",
        "nearest_ball_dist",
        "centroid_vel_fwd",
        "spread_width",
        "frac_running_fwd",
        "n_within_5m",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for ax, name in zip(axes.ravel(), picks, strict=True):
        k = TEAM_FEATURES.index(name)
        for prov, data in results.items():
            arr = data["team_change"]
            if len(arr):
                ax.plot(t, arr[:, :, k].mean(0), label=f"{prov} (n={len(arr)})")
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{name}: new − old possessor")
        ax.set_xlabel("s from possession change")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "transition_profiles.png", dpi=110)
    plt.close(fig)
    written.append("transition_profiles.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for ax, name in zip(axes, ("all_mean_speed", "ball_speed", "ball_near_line"), strict=True):
        k = GLOBAL_FEATURES.index(name)
        for prov, data in results.items():
            arr = data["dead_entry"]
            if len(arr):
                ax.plot(t, arr[:, :, k].mean(0), label=f"{prov} (n={len(arr)})")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{name} around ball going dead")
        ax.set_xlabel("s from dead-ball start")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dead_entry_profiles.png", dpi=110)
    plt.close(fig)
    written.append("dead_entry_profiles.png")
    return written


def main() -> int:
    """Run the EDA and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report", type=Path, default=ROOT / "experiments/research/neural/EDA_POSSESSION.md"
    )
    args = parser.parse_args()
    out_dir = ROOT / "artifacts" / "neural" / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "idsse": collect(TRAIN["idsse"], degrade_ball=False),
        "idsse_degraded": collect(TRAIN["idsse"], degrade_ball=True),
        "skillcorner": collect(TRAIN["skillcorner"], degrade_ball=False),
        "metrica": collect(TRAIN["metrica"], degrade_ball=False),
    }
    figs = plot_profiles({k: v for k, v in results.items() if k != "idsse_degraded"}, out_dir)

    lines = [
        "# Possession behaviour EDA — external training matches only",
        "",
        "Generated by `scripts/neural/eda_possession.py`. Every 5th frame of the *training* split "
        "matches (IDSSE 4, SkillCorner 12, Metrica game 1); validation/test matches and the "
        "challenge game are never read. Features: `src/possession_lab/neural/team_features.py` "
        "(units: m, m/s, counts, fractions; direction-relative to each team's attack).",
        "",
    ]
    for prov, data in results.items():
        y, seen = data["y"], data["seen"]
        alive = np.isin(y, (H, A))
        lines += [
            f"## {prov}",
            "",
            f"Frames analysed: {len(y):,} (alive {alive.mean():.0%}); ball unseen on "
            f"{(~seen[alive]).mean():.0%} of alive frames; transitions: team→team "
            f"{len(data['team_change'])}, alive→dead {len(data['dead_entry'])}, dead→alive {len(data['restart'])}.",
            "",
            "**Team behaviour: in possession vs defending** (AUC of home−away difference for H vs A; "
            "0.5 = useless, 1.0 = perfect; 'unseen' = only frames without a ball observation)",
            "",
            "| feature | median in possession | median defending | AUC all | AUC ball unseen |",
            "|---|---|---|---|---|",
        ]
        for name, att, dfn, a_all, a_un in team_table(data):
            lines.append(f"| {name} | {att:.2f} | {dfn:.2f} | {a_all:.3f} | {a_un:.3f} |")
        lines += [
            "",
            "**Live vs dead ball**",
            "",
            "| feature | median alive | median dead | AUC alive>dead |",
            "|---|---|---|---|",
        ]
        for name, al, de, a in dead_table(data):
            lines.append(f"| {name} | {al:.2f} | {de:.2f} | {a:.3f} |")
        lines.append("")
    lines += (
        ["## Figures (local, gitignored)", ""]
        + [f"- `artifacts/neural/eda/{f}`" for f in figs]
        + [""]
    )
    args.report.write_text("\n".join(lines), encoding="utf-8")
    np.savez_compressed(
        out_dir / "profiles.npz",
        **{
            f"{p}_{k}": v[k]
            for p, v in results.items()
            for k in ("team_change", "dead_entry", "restart")
        },
    )
    print(f"report: {args.report.relative_to(ROOT)}; figures: {', '.join(figs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
