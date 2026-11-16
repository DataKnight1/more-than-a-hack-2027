"""Generate official competition submission CSV from frozen N020 PathCRF candidate.

Usage:
    .venv-neural/Scripts/python scripts/neural/predict_n020_pathcrf.py --run n020_pathcrf_strongpaths
        [--ckpt best.pt] [--out artifacts/neural/submissions/n020_pathcrf.csv] [--device cuda]

Strict constraints:
1. Pure player trajectories alone (no match.ball input).
2. Exactly 199,319 rows with header "frame,label" and classes H, A, D, X.
3. Official submission validator check at termination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from possession_lab.cli import write_csv  # noqa: E402
from possession_lab.neural.atd import load_challenge_match  # noqa: E402
from possession_lab.neural.decode import DecodeConfig  # noqa: E402
from possession_lab.neural.pathcrf.checkpoints import load_n020_checkpoint  # noqa: E402
from possession_lab.neural.pathcrf.inference import decode_n020_match, predict_n020_probabilities  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="Run name under artifacts/neural/runs/")
    parser.add_argument("--ckpt", default="best.pt", help="Checkpoint name")
    parser.add_argument("--out", type=Path, default=None, help="Output submission CSV path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_dir = ROOT / "artifacts" / "neural" / "runs" / args.run
    ckpt_path = run_dir / args.ckpt
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, crf, state = load_n020_checkpoint(ckpt_path, device=device)
    model.eval()
    cfg = state["config"]
    sample_fps = float(cfg["train"].get("sample_fps", 5.0))

    # Read frozen decoder
    decoder_file = run_dir / "decoder_val_selected.json"
    if not decoder_file.is_file():
        decoder_file = run_dir / "external_eval.json"

    if decoder_file.is_file():
        dec_info = json.loads(decoder_file.read_text())
        if "decoder" in dec_info:
            dec_info = dec_info["decoder"]
        team_switch = float(dec_info.get("team_switch", 2.0))
        dead_switch = float(dec_info.get("dead_switch", 4.0))
        path_weight = float(dec_info.get("path_weight", 0.5))
    else:
        team_switch = float(cfg["decode"].get("team_switch", 2.0))
        dead_switch = float(cfg["decode"].get("dead_switch", 4.0))
        path_weight = float(cfg["decode"].get("path_weight", 0.5))

    decoder = DecodeConfig(team_switch=team_switch, dead_switch=dead_switch, transition_gain=0.0)

    print("================================================================================")
    print(f"N020 PATHCRF INFERENCE & SUBMISSION GENERATION: {args.run}")
    print(f"Device: {device} | Checkpoint: {ckpt_path.name}")
    print(
        f"Decoder: team_switch={team_switch}, dead_switch={dead_switch}, path_weight={path_weight}"
    )
    print("================================================================================")

    # Load challenge match (tactical ATD tracking, no GT labels)
    print("Loading target challenge fixture...", flush=True)
    match = load_challenge_match()
    assert match.n_frames == 199319, f"Expected 199,319 frames, got {match.n_frames}"

    print(
        f"Computing full-period 25 Hz probabilities across {match.n_frames:,} frames...", flush=True
    )
    probs_25hz = predict_n020_probabilities(
        model,
        match,
        sample_fps=sample_fps,
        output_fps=25.0,
        device=device,
        window_seconds=10.0,
        overlap_seconds=2.0,
        path_weight=path_weight,
    )

    print("Decoding match sequence via Viterbi transition matrix...", flush=True)
    labels = decode_n020_match(probs_25hz, match.period, decoder)
    assert len(labels) == 199319, f"Expected 199,319 labels, got {len(labels)}"

    out_csv = args.out or (ROOT / "artifacts" / "neural" / "submissions" / f"{args.run}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    digest = write_csv(out_csv, labels.tolist())
    print(f"Successfully written {len(labels):,} rows to: {out_csv}")
    print(f"File SHA-256 digest: {digest}")

    # Official submission validation
    print("\nRunning official submission validator...", flush=True)
    val_proc = subprocess.run(
        [sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(out_csv)],
        capture_output=True,
        text=True,
        check=False,
    )
    print(val_proc.stdout.strip())
    if val_proc.stderr:
        print("Validator stderr:", val_proc.stderr.strip())

    return val_proc.returncode


if __name__ == "__main__":
    sys.exit(main())
