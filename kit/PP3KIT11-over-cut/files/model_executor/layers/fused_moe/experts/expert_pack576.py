# SPDX-License-Identifier: Apache-2.0
"""Version and geometry of the TP4 packed routed-expert format."""
from __future__ import annotations

import enum
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HIDDEN = 5120
NUM_EXPERTS = 384
LAYERS = 40
RANKS = 4
NATIVE_FORMAT = "nvfp4-8r-mxfp4-v1"
PACKED_FORMAT = "nvfp4-8r-mxfp4-v3-packed576"
SWITCH = "VLLM_V41_EXPERT_PACKED576"
MANIFEST = "manifest.json"
FILE_NAME = re.compile(r"L(\d{2})-r([0-3])-(hot|cold)\.bin", re.ASCII)


class Layout(enum.Enum):
    PADDED = 640
    PACKED = 576

    @property
    def w2_k(self) -> int:
        return self.value

    @property
    def record_bytes(self) -> int:
        return 2 * self.PACKED.w2_k * HIDDEN * 17 // 32 + HIDDEN * self.w2_k * 17 // 32


def requested_layout(value: str) -> Layout:
    if value == "0":
        return Layout.PADDED
    if value == "1":
        return Layout.PACKED
    raise ValueError(f"{SWITCH} requires 0 or 1, got {value!r}")


def fields(value: object, keys: set[str]) -> dict:
    if type(value) is not dict or value.keys() != keys:
        raise ValueError(f"expected manifest fields {sorted(keys)}")
    return value


def digest(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


def natural(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"expected a nonnegative integer, got {value!r}")
    return value


def expert_ids(values: object) -> tuple[int, ...]:
    if type(values) is not list or any(
        type(x) is not int or not 0 <= x < NUM_EXPERTS for x in values
    ):
        raise ValueError("expert IDs must be a list of in-range integers")
    return tuple(values)


@dataclass(frozen=True, order=True)
class FileKey:
    layer: int
    rank: int
    tier: str

    def __post_init__(self) -> None:
        if not 0 <= self.layer < LAYERS or not 0 <= self.rank < RANKS:
            raise ValueError("file is outside the TP4 model geometry")
        if self.tier not in ("hot", "cold"):
            raise ValueError("file tier must be hot or cold")

    @property
    def name(self) -> str:
        return f"L{self.layer:02d}-r{self.rank}-{self.tier}.bin"

    @classmethod
    def parse(cls, name: object) -> FileKey:
        matched = FILE_NAME.fullmatch(name) if type(name) is str else None
        if matched is None:
            raise ValueError("invalid expert filename")
        return cls(int(matched[1]), int(matched[2]), matched[3])


@dataclass(frozen=True)
class PackedFile:
    key: FileKey
    ids: tuple[int, ...]
    sha256_before: str
    sha256_after: str

    @property
    def bytes_before(self) -> int:
        return len(self.ids) * Layout.PADDED.record_bytes

    @property
    def bytes_after(self) -> int:
        return len(self.ids) * Layout.PACKED.record_bytes

    @classmethod
    def parse(cls, value: object) -> PackedFile:
        row = fields(value, {
            "name", "expert_ids", "expert_count", "bytes_before", "bytes_after",
            "sha256_before", "sha256_after",
        })
        out = cls(FileKey.parse(row["name"]), expert_ids(row["expert_ids"]),
                  digest(row["sha256_before"]), digest(row["sha256_after"]))
        lengths = tuple(natural(row[k]) for k in ("expert_count", "bytes_before", "bytes_after"))
        if lengths != (len(out.ids), out.bytes_before, out.bytes_after):
            raise ValueError(f"{out.key.name}: wrong expert count or byte geometry")
        return out

    def document(self) -> dict:
        return {
            "name": self.key.name, "expert_ids": list(self.ids),
            "expert_count": len(self.ids), "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after, "sha256_before": self.sha256_before,
            "sha256_after": self.sha256_after,
        }


@dataclass(frozen=True)
class PackedPack:
    root: Path
    files: dict[FileKey, PackedFile]

    def __post_init__(self) -> None:
        partitions = defaultdict(dict)
        for key, row in self.files.items():
            partitions[key.layer, key.rank][key.tier] = row.ids
        if not partitions:
            raise ValueError("packed manifest has no expert files")
        for (layer, rank), groups in partitions.items():
            validate_partition(groups, layer, rank)

    @property
    def expert_count(self) -> int:
        return sum(len(row.ids) for row in self.files.values())

    def require_complete(self) -> None:
        if len(self.files) != LAYERS * RANKS * 2:
            raise ValueError("packed serving requires all 40 layers and 4 TP ranks")

    def document(self) -> dict:
        return {
            "format": PACKED_FORMAT, "expert_count": self.expert_count,
            "record_bytes_before": Layout.PADDED.record_bytes,
            "record_bytes_after": Layout.PACKED.record_bytes,
            "files": [row.document() for row in self.files.values()],
        }

    @classmethod
    def parse(cls, root: Path, value: object) -> PackedPack:
        doc = fields(value, {
            "format", "files", "expert_count", "record_bytes_before", "record_bytes_after",
        })
        if doc["format"] != PACKED_FORMAT or type(doc["files"]) is not list:
            raise ValueError(f"{root}: expected {PACKED_FORMAT}")
        for fmt, key in ((Layout.PADDED, "record_bytes_before"), (Layout.PACKED, "record_bytes_after")):
            if natural(doc[key]) != fmt.record_bytes:
                raise ValueError(f"{root}: wrong geometry for {key}")
        rows = [PackedFile.parse(row) for row in doc["files"]]
        out = cls(root, {row.key: row for row in rows})
        if len(rows) != len(out.files) or out.expert_count != natural(doc["expert_count"]):
            raise ValueError(f"{root}: duplicate file or wrong aggregate expert count")
        return out

    @classmethod
    def read(cls, root: Path) -> PackedPack:
        return cls.parse(root, json.loads((root / MANIFEST).read_bytes()))


def validate_partition(groups: dict[str, tuple[int, ...]], layer: int, rank: int) -> None:
    if groups.keys() != {"hot", "cold"} or sorted(groups["hot"] + groups["cold"]) != list(
        range(NUM_EXPERTS)
    ):
        raise ValueError(f"L{layer:02d} r{rank}: hot/cold must partition {NUM_EXPERTS} experts")
