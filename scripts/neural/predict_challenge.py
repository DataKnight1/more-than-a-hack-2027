"""Fixed local inference on the challenge tactical ATD -> 199,319-row submission CSV.

Governance: the model and decoder are frozen (trained and selected on external data only). This
script performs inference and automatic format validation; it never reads any ground truth and
prints only hashes and the official validator's summary — never tracking content.

Determinism: CPU, float32, ``torch.use_deterministic_algorithms(True)``, a fixed thread count, and
log-probabilities quantised to 1e-3 before Viterbi so last-bit float differences between machines
cannot flip a decision.

Usage::

    .venv-neural/Scripts/python scripts/neural/predict_challenge.py --run n001_set_temporal_base
        [--out artifacts/neural/submissions/n001.csv]
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from possession_lab.cli import write_csv  # noqa: E402
from possession_lab.neural.atd import load_challenge_match  # noqa: E402
from possession_lab.neural.canonical import LABEL_CHARS  # noqa: E402
from possession_lab.neural.data import predict_log_probs  # noqa: E402
from possession_lab.neural.decode import DecodeConfig, decode_match  # noqa: E402
from possession_lab.neural.model import ModelConfig, SetTemporalNet  # noqa: E402

QUANTUM = 1e-3


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """Run frozen inference and the official validator."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--ckpt", default="best.pt")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    run_dir = ROOT / "artifacts" / "neural" / "runs" / args.run
    ckpt_path = run_dir / args.ckpt
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SetTemporalNet(ModelConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"])
    model.eval()
    cfg = state["config"]
    decoder_info = json.loads((run_dir / "external_eval.json").read_text())["decoder"]
    decoder = DecodeConfig(
        decoder_info["team_switch"],
        decoder_info["dead_switch"],
        decoder_info.get("transition_gain", 0.0),
    )

    match = load_challenge_match()
    logp = predict_log_probs(
        model,
        match,
        cfg["train"]["window"],
        cfg["train"]["eval_stride"],
        torch.device("cpu"),
        cfg["train"]["max_players"],
    )
    logp = (np.round(logp / QUANTUM) * QUANTUM).astype(np.float32)
    codes = decode_match(logp, match.period, decoder)
    labels = LABEL_CHARS[codes]

    out = (args.out or ROOT / "artifacts" / "neural" / "submissions" / f"{args.run}.csv").resolve()
    digest = write_csv(out, labels.tolist())
    print(f"checkpoint sha256 {sha256(ckpt_path)}")
    print(
        f"decoder team_switch={decoder.team_switch} dead_switch={decoder.dead_switch} "
        f"transition_gain={decoder.transition_gain}"
    )
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"wrote {shown} ({len(labels)} rows) sha256 {digest}")
    result = subprocess.run(
        [sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip()[-1500:])
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
