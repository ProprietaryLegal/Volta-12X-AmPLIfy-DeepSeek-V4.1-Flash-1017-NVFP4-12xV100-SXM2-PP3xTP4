#!/usr/bin/env python3
"""Check the pinned C7 pack manifest and its 320 mounted pack files."""
import hashlib
import json
import sys
from pathlib import Path

MANIFEST_SHA = "3ae204a326efad23cd7c96429b05197a54b4d35fd4a764d3a8d90c6f494dbc69"


def packed_root(path: Path) -> Path:
    root = path.resolve(strict=True)
    manifest = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != MANIFEST_SHA:
        raise ValueError("packed-576 manifest differs from the C7 staging")
    records = json.loads(manifest)["files"]
    if len({row["name"] for row in records}) != 320:
        raise ValueError("expected 320 distinct packed-576 files")
    for row in records:
        name = Path(row["name"])
        if len(name.parts) != 1 or name.name != row["name"]:
            raise ValueError("unsafe pack filename")
        pack = root / name
        if not pack.is_file() or pack.stat().st_size != row["bytes_after"]:
            raise ValueError(f"pack missing or wrong size: {name}")
    return root


if __name__ == "__main__":
    print(packed_root(Path(sys.argv[1])))
