#!/usr/bin/env python3

from __future__ import annotations

import json

from pystamps.prep.gamma_candidates import (
    CandidateConfig,
)
from pystamps.prep.gamma_patches import (
    PatchConfig,
)
from pystamps.prep.gamma_ps_optimization import (
    PSOptimizationConfig,
)
from pystamps.prep.gamma_stage1 import (
    GammaStage1Config,
    prepare_gamma_sbas_stage1,
)


PROJECT = (
    "/mnt/vol-gdc28n1r/insar/"
    "cangzhou_P69"
)

OUTPUT = (
    "/mnt/vol-gdc28n1r/insar/"
    "cangzhou_P69/"
    "pystamps_sbas_smoke"
)


config = GammaStage1Config(
    candidate=CandidateConfig(
        # Smoke测试保留原来的较宽PS候选阈值。
        da_threshold=0.60,

        min_valid_fraction=0.90,

        # 只读取8行，块大小也设置为8行。
        block_rows=8,

        # GAMMA MLI存储功率值。
        mli_is_power=True,

        # Smoke测试不执行逐景幅度归一化，
        # 与前面的测试结果保持一致。
        normalize_per_image=False,
    ),

    patches=PatchConfig(
        # 将距离向拆成20块，每块约297列。
        range_patches=1,

        # 将方位向拆成552块，每块约10行。
        # 当前候选区域只有8行，因此实际只有少数patch有点。
        azimuth_patches=552,

        # Smoke的小patch只需要较小重叠。
        range_overlap=8,
        azimuth_overlap=2,
    ),

    ps_optimization=PSOptimizationConfig(
        # Smoke阶段不做PS限流，
        # 仅验证Stage 1到Stage 2接口。
        enabled=False,

        # enabled=False时以下空间限流参数不参与选点，
        # 但必须提供合法值。
        cell_rows=32,
        cell_cols=32,
        max_candidates_per_cell=32,
        global_max_candidates=None,
        target_candidates_per_patch=2000,

        min_range_patches=1,
        min_azimuth_patches=1,
        max_range_patches=40,
        max_azimuth_patches=600,
    ),

    # Smoke使用上面明确指定的空间patch布局。
    auto_patch_layout=False,

    reference_date=None,

    # 自动复用或生成雷达坐标经纬度。
    longitude_file=None,
    latitude_file=None,

    height_file=(
        "/mnt/vol-gdc28n1r/insar/"
        "cangzhou_P69/DEM_prep/"
        "20210211.hgt"
    ),

    range_looks=4,
    azimuth_looks=1,

    # 每个候选点最多允许1个无效干涉图。
    max_invalid_interferograms=1,

    # 小patch中至少保留50个有效PS点。
    minimum_patch_candidates=50,

    # 只测试中部连续8行。
    candidate_row_start=2750,
    candidate_row_stop=2758,

    force_lonlat=False,

    # 每次重新生成Smoke数据集。
    force=True,
)


manifest = prepare_gamma_sbas_stage1(
    PROJECT,
    OUTPUT,
    config=config,
)

print(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    )
)
