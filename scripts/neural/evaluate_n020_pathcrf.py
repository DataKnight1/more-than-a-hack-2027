"""Evaluate N020 PathCRF models on external validation and the 3-match development audit.

Usage:
    # 1. Select optimal decoder on external validation (degraded):
    .venv-neural/Scripts/python scripts/neural/evaluate_n020_pathcrf.py --run n020_pathcrf_strongpaths --split validation

    # 2. Score frozen candidate on the 3-match development audit:
    .venv-neural/Scripts/python scripts/neural/evaluate_n020_pathcrf.py --run n020_pathcrf_strongpaths --split audit --frozen

Scoring rules:
- Audit matches: idsse_J03WN1, idsse_J03WR9, metrica_3.
- Evaluated in both Clean and Degraded (seed=7) conditions.
- Primary metric: Headline mean of Clean and Degraded pooled spellF1.
- N018 champion target: 0.6919 (Promotion gate: >= 0.7019, Val > 0.5568, Dead regression <= 0.005).
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

from possession_lab.neural.canonical import CanonicalMatch  # noqa: E402
from possession_lab.neural.data import degrade_match  # noqa: E402
from possession_lab.neural.decode import DecodeConfig, decode_match  # noqa: E402
from possession_lab.neural.external_eval import pooled, score  # noqa: E402
from possession_lab.neural.pathcrf.checkpoints import load_n020_checkpoint  # noqa: E402
from possession_lab.neural.pathcrf.inference import (  # noqa: E402
    decode_n020_match,
    extract_n020_match_logits,
    fuse_and_upsample_n020_logits,
    predict_n020_probabilities,
)

TEAM_GRID = (2.0, 4.0, 8.0, 12.0, 16.0, 24.0)
DEAD_GRID = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
PATH_GRID = (0.0, 0.5)

# 3-match development audit matches
AUDIT_MATCHES = ["idsse_J03WN1", "idsse_J03WR9", "metrica_3"]


def score_candidate(
    probs_dict: dict[str, np.ndarray],
    matches: list[CanonicalMatch],
    decoder: DecodeConfig,
) -> dict[str, Any]:
    """Score decoded predictions against ground truth for a set of matches."""
    char2code = {"H": 0, "A": 1, "D": 2, "X": 3}
    per_match = {}

    for m in matches:
        probs = probs_dict[m.match_id]
        labels_str = decode_n020_match(probs, m.period, decoder)
        pred_codes = np.array([char2code[c] for c in labels_str], dtype=np.int8)
        per_match[m.match_id] = score(m.labels, pred_codes)

    return {
        "pooled": pooled(per_match),
        "per_match": per_match,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run", required=True, help="Run directory name under artifacts/neural/runs/"
    )
    parser.add_argument("--ckpt", default="best.pt", help="Checkpoint filename (default: best.pt)")
    parser.add_argument(
        "--split", choices=["validation", "audit"], required=True, help="Evaluation target split"
    )
    parser.add_argument(
        "--frozen", action="store_true", help="Audit mode: use frozen validation-selected decoder"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = ROOT / "artifacts" / "neural" / "runs" / args.run
    ckpt_path = run_dir / args.ckpt
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)
    model, crf, state = load_n020_checkpoint(ckpt_path, device=device)
    cfg = state["config"]
    canon_dir = ROOT / cfg["data"]["canonical_dir"]
    sample_fps = float(cfg["train"].get("sample_fps", 5.0))
    degrade_seed = int(cfg["data"].get("degrade_eval_seed", 7))

    print("================================================================================")
    print(f"N020 PATHCRF EVALUATION: {args.run} ({args.ckpt})")
    print(f"Split: {args.split.upper()} | Sample FPS: {sample_fps} | Device: {device}")
    print("================================================================================")

    # --------------------------------------------------------------------------
    # SPLIT: VALIDATION
    # --------------------------------------------------------------------------
    if args.split == "validation":
        val_names = cfg["data"]["val"]
        print(f"Loading {len(val_names)} validation matches: {val_names}...", flush=True)
        val_clean = [CanonicalMatch.load(canon_dir / f"{n}.npz") for n in val_names]
        val_degraded = [degrade_match(m, degrade_seed) for m in val_clean]
        all_val = val_clean + val_degraded

        # Precompute match logits once across all validation matches
        print(f"Extracting neural logits for {len(all_val)} validation matches...", flush=True)
        raw_match_logits = []
        for m in all_val:
            raw_match_logits.append(
                extract_n020_match_logits(
                    model,
                    m,
                    sample_fps=sample_fps,
                    device=device,
                    window_seconds=10.0,
                    overlap_seconds=2.0,
                )
            )

        best_overall_score = -1.0
        best_cfg = (2.0, 4.0, 0.5)

        for pw in PATH_GRID:
            print(f"Fusing probabilities for path_weight={pw}...", flush=True)
            val_lps = []
            for m, rml in zip(all_val, raw_match_logits, strict=True):
                p_25 = fuse_and_upsample_n020_logits(
                    rml,
                    total_frames=m.n_frames,
                    sample_fps=sample_fps,
                    model=model,
                    path_weight=pw,
                    device=device,
                )
                val_lps.append(np.log(np.clip(p_25, 1e-7, 1.0)).astype(np.float32))

            # Grid search team & dead penalties
            combos = list(itertools.product(TEAM_GRID, DEAD_GRID))
            for i, (t_pen, d_pen) in enumerate(combos):
                dec = DecodeConfig(team_switch=t_pen, dead_switch=d_pen, transition_gain=0.0)
                scores = [
                    score(m.labels, decode_match(lp, m.period, dec))["spell_f1_mean"]
                    for m, lp in zip(all_val, val_lps, strict=True)
                ]
                val_s = float(np.mean(scores))
                if (i + 1) % 6 == 0 or (i + 1) == len(combos):
                    print(
                        f"  [pw={pw}] Grid {i + 1:2d}/{len(combos)}: team={t_pen:4.1f}, dead={d_pen:4.1f} -> val_s={val_s:.4f} (best so far: {max(best_overall_score, val_s):.4f})",
                        flush=True,
                    )
                if val_s > best_overall_score:
                    best_overall_score = val_s
                    best_cfg = (t_pen, d_pen, pw)

        best_team, best_dead, best_path = best_cfg
        print("--------------------------------------------------------------------------------")
        print(
            f"DECODER SELECTION COMPLETE ON VALIDATION:\n"
            f"  team_switch: {best_team}\n"
            f"  dead_switch: {best_dead}\n"
            f"  path_weight: {best_path}\n"
            f"  val_spellF1: {best_overall_score:.4f}"
        )
        print("--------------------------------------------------------------------------------")

        val_selection = {
            "team_switch": best_team,
            "dead_switch": best_dead,
            "path_weight": best_path,
            "val_spell_f1": best_overall_score,
            "val_matches": val_names,
        }
        (run_dir / "decoder_val_selected.json").write_text(json.dumps(val_selection, indent=2))
        print(f"Written: {run_dir / 'decoder_val_selected.json'}")
        return 0

    # --------------------------------------------------------------------------
    # SPLIT: 3-MATCH DEVELOPMENT AUDIT
    # --------------------------------------------------------------------------
    print(
        "Evaluating 3-Match Development Audit (idsse_J03WN1, idsse_J03WR9, metrica_3)...",
        flush=True,
    )
    val_sel_file = run_dir / "decoder_val_selected.json"
    if args.frozen and val_sel_file.is_file():
        v_sel = json.loads(val_sel_file.read_text())
        team_switch = float(v_sel["team_switch"])
        dead_switch = float(v_sel["dead_switch"])
        path_weight = float(v_sel["path_weight"])
        val_f1_logged = float(v_sel["val_spell_f1"])
        print(
            f"Loaded frozen decoder: team={team_switch}, dead={dead_switch}, path_w={path_weight} (val spellF1: {val_f1_logged:.4f})"
        )
    else:
        team_switch = float(cfg["decode"].get("team_switch", 2.0))
        dead_switch = float(cfg["decode"].get("dead_switch", 4.0))
        path_weight = float(cfg["decode"].get("path_weight", 0.5))
        val_f1_logged = float(state.get("val_spell_f1", 0.0))
        print(
            f"Using default config decoder: team={team_switch}, dead={dead_switch}, path_w={path_weight}"
        )

    decoder = DecodeConfig(team_switch=team_switch, dead_switch=dead_switch, transition_gain=0.0)

    clean_matches = [CanonicalMatch.load(canon_dir / f"{n}.npz") for n in AUDIT_MATCHES]
    degraded_matches = [degrade_match(m, degrade_seed) for m in clean_matches]

    report: dict[str, Any] = {
        "run": args.run,
        "checkpoint": args.ckpt,
        "decoder": {
            "team_switch": team_switch,
            "dead_switch": dead_switch,
            "path_weight": path_weight,
            "val_spell_f1_logged": val_f1_logged,
        },
        "test": {},
    }

    # Evaluate Clean condition
    print("\n[CLEAN CONDITION AUDIT]", flush=True)
    clean_probs = {
        m.match_id: predict_n020_probabilities(
            model, m, sample_fps=sample_fps, output_fps=25.0, device=device, path_weight=path_weight
        )
        for m in clean_matches
    }
    clean_results = score_candidate(clean_probs, clean_matches, decoder)
    report["test"]["clean"] = {"neural_viterbi": clean_results}

    clean_spell = clean_results["pooled"]["spell_f1_mean"]
    clean_macro = clean_results["pooled"]["macro_f1"]
    clean_ad = clean_results["pooled"]["alive_dead_f1"]
    print(
        f"Clean Pooled: SpellF1: {clean_spell:.4f} | MacroF1: {clean_macro:.4f} | Dead/Alive: {clean_ad:.4f}"
    )

    # Evaluate Degraded condition
    print("\n[DEGRADED CONDITION AUDIT]", flush=True)
    deg_probs = {
        m.match_id: predict_n020_probabilities(
            model, m, sample_fps=sample_fps, output_fps=25.0, device=device, path_weight=path_weight
        )
        for m in degraded_matches
    }
    deg_results = score_candidate(deg_probs, degraded_matches, decoder)
    report["test"]["degraded"] = {"neural_viterbi": deg_results}

    deg_spell = deg_results["pooled"]["spell_f1_mean"]
    deg_macro = deg_results["pooled"]["macro_f1"]
    deg_ad = deg_results["pooled"]["alive_dead_f1"]
    print(
        f"Degraded Pooled: SpellF1: {deg_spell:.4f} | MacroF1: {deg_macro:.4f} | Dead/Alive: {deg_ad:.4f}"
    )

    # Headline Summary
    headline_mean_spell = (clean_spell + deg_spell) / 2.0
    headline_mean_macro = (clean_macro + deg_macro) / 2.0
    headline_mean_ad = (clean_ad + deg_ad) / 2.0

    report["headline"] = {
        "mean_spell_f1": headline_mean_spell,
        "mean_macro_f1": headline_mean_macro,
        "mean_alive_dead_f1": headline_mean_ad,
    }

    eval_out = run_dir / "external_eval.json"
    eval_out.write_text(json.dumps(report, indent=2))
    print(f"\nWritten complete evaluation to: {eval_out}")

    # Comparison against N018 Champion
    n018_target_spell = 0.6919
    n018_clean_spell = 0.7673
    n018_deg_spell = 0.6166
    n018_ad_target = 0.9679
    delta_spell = headline_mean_spell - n018_target_spell

    print("================================================================================")
    print("AUDIT PROMOTION GATE VERIFICATION")
    print("================================================================================")
    print("Metric                       N018 Champion    N020 Candidate    Delta")
    print("--------------------------------------------------------------------------------")
    print(
        f"Headline Audit spellF1       {n018_target_spell:.4f}           {headline_mean_spell:.4f}          {delta_spell:+.4f}"
    )
    print(
        f"  - Clean condition          {n018_clean_spell:.4f}           {clean_spell:.4f}          {clean_spell - n018_clean_spell:+.4f}"
    )
    print(
        f"  - Degraded condition       {n018_deg_spell:.4f}           {deg_spell:.4f}          {deg_spell - n018_deg_spell:+.4f}"
    )
    print(
        f"External Validation spellF1  0.5568           {val_f1_logged:.4f}          {val_f1_logged - 0.5568:+.4f}"
    )
    print(
        f"Alive / Dead F1              {n018_ad_target:.4f}           {headline_mean_ad:.4f}          {headline_mean_ad - n018_ad_target:+.4f}"
    )
    print("--------------------------------------------------------------------------------")

    promoted = (
        headline_mean_spell >= 0.7019
        and val_f1_logged > 0.5568
        and (headline_mean_ad - n018_ad_target) >= -0.005
    )
    if promoted:
        print("VERDICT: PROMOTION CRITERIA SATISFIED! N020 BEATS N018 CHAMPION.")
    else:
        print("VERDICT: PROMOTION CRITERIA NOT MET. PRESERVE N018 AS CHAMPION.")
    print("================================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
