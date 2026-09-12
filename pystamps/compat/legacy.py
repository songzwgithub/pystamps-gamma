from __future__ import annotations

from pathlib import Path


def discover_legacy_commands(stamps_root: str | Path = "StaMPS") -> list[Path]:
    bin_dir = Path(stamps_root) / "bin"
    if not bin_dir.exists():
        return []
    return sorted([p for p in bin_dir.iterdir() if p.is_file()])
