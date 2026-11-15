"""Rebuild the large NASA rasters from the compressed parts shipped in the GitHub/release package.

GitHub refuses files > 100 MB (and > 25 MB through the browser upload), so `tools/build_release.py` stores each
big raster as gzip parts of <= 24 MB under `data/packed/` plus a manifest with the original size and SHA-256.
`ensure_data()` runs on import of `clps_engine` and, only when an original is missing, joins + decompresses the
parts back to the exact original path and verifies the checksum. When the originals are present it does nothing.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKED = ROOT / "data" / "packed"
MANIFEST = PACKED / "manifest.json"


class _PartsReader:
    """File-like object reading a list of part files back to back."""

    def __init__(self, parts):
        self._parts = list(parts)
        self._fh = None

    def read(self, n=-1):
        out = bytearray()
        while n < 0 or len(out) < n:
            if self._fh is None:
                if not self._parts:
                    break
                self._fh = open(self._parts.pop(0), "rb")
            chunk = self._fh.read(-1 if n < 0 else n - len(out))
            if not chunk:
                self._fh.close()
                self._fh = None
                continue
            out += chunk
        return bytes(out)


def _ok(path: Path, size: int) -> bool:
    return path.exists() and path.stat().st_size == size


def ensure_data(verbose: bool = True) -> list[str]:
    """Restore any missing packed file. Returns the list of files rebuilt."""
    if not MANIFEST.exists() or os.environ.get("CLPS_SKIP_BOOTSTRAP"):
        return []
    rebuilt = []
    for entry in json.loads(MANIFEST.read_text())["files"]:
        target = ROOT / entry["path"]
        if _ok(target, entry["size"]):
            continue
        parts = [PACKED / p for p in entry["parts"]]
        missing = [p.name for p in parts if not p.exists()]
        if missing:
            raise FileNotFoundError(f"data/packed is incomplete for {entry['path']}: missing {missing}")
        if verbose:
            print(f"[LunarSouth] unpacking {entry['path']} ({entry['size'] / 2**20:.0f} MB, first run only) ...", flush=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        h = hashlib.sha256()
        with gzip.GzipFile(fileobj=_PartsReader(parts)) as src, open(tmp, "wb") as dst:
            while True:
                buf = src.read(4 << 20)
                if not buf:
                    break
                h.update(buf)
                dst.write(buf)
        if h.hexdigest() != entry["sha256"] or tmp.stat().st_size != entry["size"]:
            tmp.unlink(missing_ok=True)
            raise IOError(f"checksum mismatch rebuilding {entry['path']} - re-download data/packed")
        tmp.replace(target)
        rebuilt.append(entry["path"])
    return rebuilt


if __name__ == "__main__":
    done = ensure_data()
    print("rebuilt:", done or "nothing (all data present)")
