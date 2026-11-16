"""Download the IDSSE / DFL open tracking + event dataset (CC BY 4.0) with MD5 verification.

Source: Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event
data in elite soccer", Scientific Data. figshare DOI 10.6084/m9.figshare.28196177 (version 1).
Licence: CC BY 4.0 (checked via the figshare API at download time and recorded in the manifest).

The files are external, other-game data. They are stored under ``data/external/idsse/raw`` which
is gitignored by ``data/*``. Standard library only so it runs in any environment.

Usage::

    python scripts/neural/fetch_idsse.py [--dest data/external/idsse/raw] [--kinds positions events]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ARTICLE_ID = 28196177
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
EXPECTED_LICENCE = "CC BY"
ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "possession-lab-research/0.1"
KIND_PREFIX = {
    "matchinformation": "DFL_02_01_matchinformation",
    "events": "DFL_03_02_events_raw",
    "positions": "DFL_04_03_positions_raw_observed",
}


def _get_json(url: str) -> dict[str, object]:
    """Fetch and decode a JSON document."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URL
        payload: dict[str, object] = json.loads(response.read().decode("utf-8"))
    return payload


def md5_of(path: Path) -> str:
    """Return the hex MD5 of a file, streamed in 8 MiB chunks."""
    digest = hashlib.md5()  # noqa: S324 - integrity check against the publisher's MD5, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    """Stream a URL to ``target`` via a temporary ``.part`` file."""
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:  # noqa: S310
        for chunk in iter(lambda: response.read(8 << 20), b""):
            out.write(chunk)
    partial.replace(target)


def main() -> int:
    """Download, verify and record the requested IDSSE file kinds."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=ROOT / "data" / "external" / "idsse" / "raw")
    parser.add_argument(
        "--kinds", nargs="+", default=sorted(KIND_PREFIX), choices=sorted(KIND_PREFIX)
    )
    args = parser.parse_args()

    article = _get_json(API_URL)
    licence = article["license"]
    if not isinstance(licence, dict) or licence.get("name") != EXPECTED_LICENCE:
        print(f"STOP: unexpected licence {licence!r}; review before use.")
        return 2
    files = article["files"]
    if not isinstance(files, list):
        raise TypeError("figshare API returned no file list")

    args.dest.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(KIND_PREFIX[kind] for kind in args.kinds)
    records: list[dict[str, object]] = []
    for entry in files:
        name = str(entry["name"])
        if not name.startswith(prefixes):
            continue
        target = args.dest / name
        expected = str(entry["computed_md5"])
        if target.is_file() and md5_of(target) == expected:
            status = "cached"
        else:
            print(f"downloading {name} ({int(entry['size']) / 1e6:.0f} MB)", flush=True)
            download(str(entry["download_url"]), target)
            if md5_of(target) != expected:
                print(f"FAIL: MD5 mismatch for {name}")
                return 1
            status = "downloaded"
        records.append({"name": name, "size": entry["size"], "md5": expected, "status": status})
        print(f"  ok {status:<10} {name}", flush=True)

    manifest = {
        "dataset": str(article["title"]),
        "doi": article.get("doi"),
        "version": article.get("version"),
        "licence": licence,
        "citation": "Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of "
        "spatiotemporal and event data in elite soccer. Scientific Data 12, 195.",
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": records,
    }
    (args.dest.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"{len(records)} files verified; manifest at {args.dest.parent / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
