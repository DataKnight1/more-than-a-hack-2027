"""Where does a model lose spellF1? Error analysis on EXTERNAL VALIDATION matches only.

Two parts:

1. *Metric sensitivity on ground truth alone* (no model): how much spellF1 a perfect-but-coarse
   prediction would get if it dropped short spells, merged short dead-ball breaks or placed
   boundaries a little late. This shows which kinds of error the metric punishes most.
2. *Model errors*: every ground-truth spell is matched to the same-team prediction with the best
   IoU; missed spells (IoU < 0.5) are classified (taken by the other team, predicted dead, merged
   into a longer prediction, split / mistimed) and broken down by duration and by how the spell
   began (restart after dead ball vs turnover).

Test matches and the challenge game are never read. Output: aggregate tables in
``experiments/research/neural/ERROR_ANALYSIS.md``; log-probabilities cached in the run directory.

Usage::

    python scripts/neural/error_analysis.py --run n008_n007_seed2
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "neural"))

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from evaluate import load_model  # noqa: E402

from possession_lab.neural.canonical import A, CanonicalMatch, D, H, X  # noqa: E402
from possession_lab.neural.data import degrade_match, predict_log_probs  # noqa: E402
from possession_lab.neural.decode import DecodeConfig, decode_match  # noqa: E402
from possession_lab.neural.external_eval import official_metrics  # noqa: E402

FPS = 25
DUR_BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]


def spell_f1(gt: np.ndarray, pred: np.ndarray) -> float:
    """Official mean spellF1 (IoU 0.3..0.7) of ``pred`` against ``gt``."""
    m = official_metrics()
    runs = m.valid_runs(gt)
    return float(m.spell_scores(gt, m.fill_abstentions(pred, runs), runs)["mean_f1"])


def segs(codes: np.ndarray, gt: np.ndarray) -> list[tuple[int, int, int]]:
    """Return maximal half-open label segments and their class codes."""
    m = official_metrics()
    return m.segments(codes, m.valid_runs(gt))


def drop_short(gt: np.ndarray, labels: tuple[int, ...], min_frames: int) -> np.ndarray:
    """Relabel segments of ``labels`` shorter than ``min_frames`` with the preceding segment."""
    out = gt.copy()
    prev = None
    for a, b, k in segs(gt, gt):
        if k in labels and b - a < min_frames and prev is not None:
            out[a:b] = prev
        else:
            prev = out[a]
    return out


def delay_boundaries(gt: np.ndarray, frames: int) -> np.ndarray:
    """Every label change happens ``frames`` later (the previous label persists)."""
    out = gt.copy()
    for a, b, _ in segs(gt, gt):
        if a > 0 and gt[a - 1] != X:
            out[a : min(b, a + frames)] = gt[a - 1]
    return out


def sensitivity(gts: list[np.ndarray]) -> list[tuple[str, float]]:
    """Measure temporal boundary matching within the configured frame tolerance."""
    rows = []
    for k in (1, 2, 3, 5):
        rows.append(
            (
                f"drop possession spells < {k} s",
                np.mean([spell_f1(g, drop_short(g, (H, A), k * FPS)) for g in gts]),
            )
        )
    for k in (1, 2, 3):
        rows.append(
            (
                f"merge dead-ball breaks < {k} s",
                np.mean([spell_f1(g, drop_short(g, (D,), k * FPS)) for g in gts]),
            )
        )
    for f in (5, 12, 25, 50):
        rows.append(
            (
                f"all boundaries {f / FPS:.1f} s late",
                np.mean([spell_f1(g, delay_boundaries(g, f)) for g in gts]),
            )
        )
    return rows


def classify(gt: np.ndarray, pred: np.ndarray) -> list[dict[str, object]]:
    """One record per ground-truth possession spell."""
    m = official_metrics()
    runs = m.valid_runs(gt)
    pf = m.fill_abstentions(pred, runs)
    gs = m.possession_spells(gt, runs)
    ps = m.possession_spells(pf, runs)
    starts = np.array([p[0] for p in ps])
    out = []
    for a, b, k in gs:
        dur = (b - a) / FPS
        before = int(gt[a - 1]) if a > 0 else X
        origin = "restart" if before == D else "turnover" if before in (H, A) else "half start"
        best, best_len = 0.0, 0
        lo = max(0, int(np.searchsorted(starts, a)) - 3)
        for pa, pb, pk in ps[lo:]:
            if pa >= b:
                break
            if pk != k or pb <= a:
                continue
            inter = min(b, pb) - max(a, pa)
            iou = inter / (max(b, pb) - min(a, pa))
            if iou > best:
                best, best_len = iou, pb - pa
        window = pf[a:b]
        share_same = float((window == k).mean())
        share_dead = float((window == D).mean())
        if best >= 0.5:
            kind = "matched"
        elif share_dead >= 0.5:
            kind = "predicted dead"
        elif share_same < 0.5:
            kind = "taken by other team"
        elif best_len > (b - a):
            kind = "merged into longer prediction"
        else:
            kind = "split / mistimed"
        out.append({"dur": dur, "origin": origin, "kind": kind, "iou": best})
    return out


def pred_spell_stats(gt: np.ndarray, pred: np.ndarray) -> list[float]:
    """Summarize predicted spell lengths and transition errors for one match."""
    m = official_metrics()
    runs = m.valid_runs(gt)
    return [(b - a) / FPS for a, b, _ in m.possession_spells(m.fill_abstentions(pred, runs), runs)]


def main() -> int:
    """Run the individual-player EDA on the configured external training split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "experiments/research/neural/ERROR_ANALYSIS.md"
    )
    args = parser.parse_args()
    args.report = args.report.resolve()
    run_dir = ROOT / "artifacts" / "neural" / "runs" / args.run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir / "best.pt", device)
    dec = json.loads((run_dir / "external_eval.json").read_text())["decoder"]
    decoder = DecodeConfig(dec["team_switch"], dec["dead_switch"], dec.get("transition_gain", 0.0))
    canon = ROOT / cfg["data"]["canonical_dir"]
    tcfg = cfg["train"]
    names = cfg["data"]["val"]
    clean = [CanonicalMatch.load(canon / f"{n}.npz") for n in names]
    conds = {
        "clean": clean,
        "degraded": [degrade_match(mm, cfg["data"]["degrade_eval_seed"]) for mm in clean],
    }

    lines = [
        f"# Error analysis — `{args.run}` on external validation matches",
        "",
        f"Validation matches: {', '.join(names)}. Decoder team {decoder.team_switch} / dead "
        f"{decoder.dead_switch} (selected on this validation set). Test matches and the "
        "challenge game are not read. Generated by `scripts/neural/error_analysis.py`.",
        "",
        "## 1. Metric sensitivity (ground truth vs a corrupted copy of itself)",
        "",
        "| perfect labels except … | spellF1 |",
        "|---|---|",
    ]
    for name, val in sensitivity([mm.labels for mm in clean]):
        lines.append(f"| {name} | {val:.3f} |")
    cache = {}
    for cond, matches in conds.items():
        recs, pred_durs, gt_durs, f1s = [], [], [], []
        for mm in matches:
            lp = predict_log_probs(
                model, mm, tcfg["window"], tcfg["eval_stride"], device, tcfg["max_players"]
            )
            cache[f"{cond}:{mm.match_id}"] = lp
            pred = decode_match(lp, mm.period, decoder)
            recs += classify(mm.labels, pred)
            pred_durs += pred_spell_stats(mm.labels, pred)
            gt_durs += pred_spell_stats(mm.labels, mm.labels)
            f1s.append(spell_f1(mm.labels, pred))
        lines += [
            "",
            f"## 2. Model errors — {cond}",
            "",
            f"Mean spellF1 {np.mean(f1s):.3f}; ground-truth spells {len(gt_durs)}, predicted "
            f"{len(pred_durs)}; median duration GT {np.median(gt_durs):.1f} s, predicted "
            f"{np.median(pred_durs):.1f} s.",
            "",
            "**Ground-truth spells by outcome at IoU 0.5**",
            "",
            "| outcome | share of spells | median duration s |",
            "|---|---|---|",
        ]
        by_kind = defaultdict(list)
        for r in recs:
            by_kind[r["kind"]].append(r["dur"])
        for kind, durs in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| {kind} | {len(durs) / len(recs):.1%} | {np.median(durs):.1f} |")
        lines += [
            "",
            "**Recall at IoU 0.5 by duration and by how the spell began**",
            "",
            "| duration | n | recall | restart n / recall | turnover n / recall | main miss |",
            "|---|---|---|---|---|---|",
        ]
        for lo, hi in DUR_BINS:
            sel = [r for r in recs if lo <= r["dur"] < hi]
            if not sel:
                continue
            rec = np.mean([r["kind"] == "matched" for r in sel])
            parts = []
            for origin in ("restart", "turnover"):
                o = [r for r in sel if r["origin"] == origin]
                parts.append(
                    f"{len(o)} / {np.mean([r['kind'] == 'matched' for r in o]):.2f}"
                    if o
                    else "0 / –"
                )
            miss = Counter(r["kind"] for r in sel if r["kind"] != "matched").most_common(1)
            label = f"{miss[0][0]} ({miss[0][1]})" if miss else "–"
            span = f"{lo}–{hi:g} s" if hi < 1e8 else f"≥ {lo} s"
            lines.append(f"| {span} | {len(sel)} | {rec:.2f} | {parts[0]} | {parts[1]} | {label} |")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savez_compressed(run_dir / "val_logp.npz", **cache)
    print(f"report: {args.report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
