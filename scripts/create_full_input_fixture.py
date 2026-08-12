#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REQUIRED_STAGE1_META = ("day.1.in", "master_day.1.in", "bperp.1.in")


def _resolve_meta_source(metadata_root: Path, patch_name: str, filename: str) -> Path | None:
    candidates = (
        metadata_root / patch_name / filename,
        metadata_root / filename,
    )
    for c in candidates:
        if c.exists():
            return c
    return None


def _can_synthesize_stage1_metadata(dataset_root: Path) -> bool:
    return any((dataset_root / "diff0").glob("*.base")) and any((dataset_root / "rslc").glob("*.rslc.par"))


def build_fixture(source: Path, dest: Path, metadata_root: Path | None, overwrite: bool) -> None:
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")
    if dest.exists():
        if not overwrite:
            raise SystemExit(f"Destination already exists (use --overwrite): {dest}")
        shutil.rmtree(dest)

    shutil.copytree(source, dest)

    missing: list[str] = []
    for patch_dir in sorted(dest.glob("PATCH_*")):
        if not patch_dir.is_dir():
            continue
        for name in REQUIRED_STAGE1_META:
            target = patch_dir / name
            if target.exists():
                continue
            if metadata_root is None:
                missing.append(f"{patch_dir.name}/{name}")
                continue
            src = _resolve_meta_source(metadata_root, patch_dir.name, name)
            if src is None:
                missing.append(f"{patch_dir.name}/{name}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)

    if missing and not _can_synthesize_stage1_metadata(dest):
        preview = ", ".join(missing[:12])
        suffix = "" if len(missing) <= 12 else f" ... (+{len(missing)-12} more)"
        raise SystemExit(
            "Fixture is missing required stage-1 metadata after copy. "
            f"Missing: {preview}{suffix}. "
            "Provide --metadata-root containing day.1.in/master_day.1.in/bperp.1.in "
            "or copy a SNAP-prepared stack with diff0/*.base and rslc/*.rslc.par."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a full-input pySTAMPS benchmark fixture.")
    parser.add_argument("--source", required=True, help="Source dataset path")
    parser.add_argument("--dest", required=True, help="Destination fixture path")
    parser.add_argument("--metadata-root", default=None, help="Optional root to copy missing stage-1 metadata from")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it already exists")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    metadata_root = Path(args.metadata_root).expanduser().resolve() if args.metadata_root else None
    build_fixture(source, dest, metadata_root, overwrite=args.overwrite)
    print(f"Fixture ready: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
