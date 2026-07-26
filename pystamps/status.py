from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pystamps.io.dataset import discover_dataset, infer_merged_stage, infer_patch_stage


@dataclass(slots=True)
class PatchStatus:
    patch: str
    stage: int


@dataclass(slots=True)
class DatasetStatus:
    dataset: Path
    patch_statuses: list[PatchStatus]
    merged_stage: int


def collect_status(dataset_root: str | Path) -> DatasetStatus:
    layout = discover_dataset(dataset_root)
    patch_statuses = [PatchStatus(patch=p.name, stage=infer_patch_stage(p)) for p in layout.patches]
    merged_stage = infer_merged_stage(layout.root)
    return DatasetStatus(dataset=layout.root, patch_statuses=patch_statuses, merged_stage=merged_stage)
