"""Tests validating format compliance of N020 generated predictions."""

import subprocess
import sys


def test_submission_csv_format(tmp_path):
    csv_path = tmp_path / "test_submission.csv"

    # Write a small valid submission file of 100 frames
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("frame,label\n")
        for i in range(100):
            lbl = "H" if i < 40 else ("A" if i < 80 else "D")
            f.write(f"{i},{lbl}\n")

    # Call validate_submission.py
    cmd = [
        sys.executable,
        "evaluation/validate_submission.py",
        str(csv_path),
        "--rows",
        "100",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Validator failed: {res.stderr}\n{res.stdout}"
    assert "OK" in res.stdout
