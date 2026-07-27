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
    "pystamps_sbas_ps_optimized"
)


config = GammaStage1Config(
    candidate=CandidateConfig(
        # 纯PS的D_A初筛。
        # 比0.60更严格，但空间限流仍会保留均匀分布。
        da_threshold=0.45,

        min_valid_fraction=0.90,

        # 控制逐块读取MLI时的临时内存。
        block_rows=128,

        # GAMMA MLI是功率，内部使用sqrt(MLI)作为幅度。
        mli_is_power=True,

        # 正式处理启用逐景幅度归一化。
        normalize_per_image=True,
    ),

    # auto_patch_layout=True时，下面的patch数量仅作为基础值；
    # overlap参数仍会沿用。
    patches=PatchConfig(
        range_patches=1,
        azimuth_patches=1,
        range_overlap=50,
        azimuth_overlap=100,
    ),

    ps_optimization=PSOptimizationConfig(
        enabled=True,

        # 32×32个4:1多视像元作为一个空间均衡单元。
        cell_rows=32,
        cell_cols=32,

        # 每个单元最多保留32个D_A最低的纯PS候选点。
        max_candidates_per_cell=32,

        # 全景最多保留100万个候选点。
        # 限制的是Stage 2输入规模，不引入DS。
        global_max_candidates=1_000_000,

        # 自动patch目标：每patch约2.5万个候选点。
        target_candidates_per_patch=50_000,

        min_range_patches=1,
        min_azimuth_patches=1,
        max_range_patches=20,
        max_azimuth_patches=20,
    ),

    auto_patch_layout=True,

    reference_date=None,

    longitude_file=None,
    latitude_file=None,

    height_file=(
        "/mnt/vol-gdc28n1r/insar/"
        "cangzhou_P69/DEM_prep/"
        "20210211.hgt"
    ),

    range_looks=4,
    azimuth_looks=1,

    max_invalid_interferograms=1,

    # 正式patch至少需要较多候选点。
    minimum_patch_candidates=500,

    candidate_row_start=0,
    candidate_row_stop=None,

    force_lonlat=False,

    # 输出到新目录，可安全重新生成。
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
