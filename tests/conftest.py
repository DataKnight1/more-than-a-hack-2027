"""Allow synthetic tests before editable package installation completes.

The ``bundle`` fixture writes a tiny fake match in the organiser's file layout (no licensed data):
a home player dribbles for 60 s from kickoff, then an away player for 60 s.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KICKOFF, HALF = 11_185, 1_500
SLOTS = [0.48, 0.30, 0.20, 0.35, 0.25, 0.52, 0.70, 0.80, 0.65, 0.75]  # x of 5 home, 5 away
META = """<Metadata><Teams><Team id="T1"><Name>FC Barcelona</Name></Team>
<Team id="T2"><Name>Athletic Club</Name></Team></Teams><Players>{}</Players></Metadata>"""
TAGS = """<file><ALL_INSTANCES><instance><ID>1</ID><start>452</start><end>487</end>
<code>BUILD UP</code><label><group>Team</group><text>Barcelona</text></label></instance>
</ALL_INSTANCES></file>"""


def raw_line(frame: int) -> str:
    """One ATD record: the ball 0.5 m ahead of the carrier, lost 2 s in every 10 s."""
    t = frame - KICKOFF
    carrier = 0 if t < HALF else 5
    groups, ball = [], "NaN,NaN,NaN"
    for slot, x in enumerate(SLOTS):
        px, py = x + (0.0002 * (t % HALF) if slot == carrier else 0.0), 0.5 + 0.01 * slot
        groups.append(f"{px:.5f},{py:.5f},2.5")
        if slot == carrier and t % 250 < 200:
            ball = f"{px + 0.5 / 105.3:.5f},{py:.5f},0.0"
    return f"{frame + 1}:{';'.join(groups)}:{ball}"


@pytest.fixture(name="bundle")
def fake_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Fake bundle under tmp_path; ``games`` and the parse cache are pointed at it."""
    folder = tmp_path / "tactical"
    folder.mkdir()
    players = "".join(f'<Player teamId="T{1 + i // 5}"/>' for i in range(len(SLOTS)))
    (folder / "BAR-ATH_tactical_atd.xml").write_text(META.format(players))
    lines = (raw_line(frame) for frame in range(KICKOFF, KICKOFF + 2 * HALF))
    (folder / "BAR-ATH_tactical_atd.txt").write_text("\n".join(lines) + "\n")
    (folder / "BAR-ATH_tactical_smart_tagging.xml").write_text(TAGS)
    monkeypatch.setattr(importlib.import_module("games"), "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(importlib.import_module("possession_lab.hmm"), "CACHE_DIR", tmp_path / "c")
    return SimpleNamespace(root=tmp_path, kickoff=KICKOFF, half=HALF)
