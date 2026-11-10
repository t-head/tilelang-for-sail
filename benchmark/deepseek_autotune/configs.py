"""Tunable parameter configuration generators for DeepSeek kernels.

Each `get_*_configs()` function returns a list of dicts whose keys correspond
to the tunable keyword arguments of the associated kernel function.
"""

import itertools
import os

import tilelang


def _pass_config_key(name):
    """Look up a PassConfigKey member by name, or None if absent.

    PPU-only keys (e.g. TL_DISABLE_AIU_LOWER, TL_DISABLE_LDMAT_SWZL) do not
    exist in CUDA-only tilelang builds; referencing them via getattr keeps
    this module importable on non-PPU targets such as H200.
    """
    return getattr(tilelang.PassConfigKey, name, None)


def _expand_pass_config_tuning(configs, dimensions):
    """Expand each config with per-config pass_configs variants.

    dimensions is a list of (PassConfigKey, values) pairs; the Cartesian
    product of all values is merged into a single per-config pass_configs
    dict, so multiple pass-config dimensions compose instead of overwriting
    each other. Only applied on PPU 1.5; on PPU 1.0 or non-PPU targets the
    original configs are returned unchanged.

    PPU 1.5 compute_version is (1, 5).
    """
    if not dimensions:
        return configs

    try:
        from tilelang.contrib import hgcc

        _arch = hgcc.get_target_compute_version()
        _compute_version = hgcc.parse_compute_version(_arch)
        if _compute_version != (1, 5):
            return configs
    except Exception:
        return configs

    expanded = []
    for cfg in configs:
        for values in itertools.product(*(vals for _, vals in dimensions)):
            new_cfg = dict(cfg)
            new_cfg["pass_configs"] = {key: val for (key, _), val in zip(dimensions, values)}
            expanded.append(new_cfg)
    return expanded


def _tuning_dimensions():
    """Collect the pass-config tuning dimensions for this run.

    Always includes TL_DISABLE_AIU_LOWER. Set TILELANG_TUNE_LDMAT_SWZL=1 to
    additionally expand over TL_DISABLE_LDMAT_SWZL (doubles the config
    space); any other value keeps the default space. Keys absent from the
    current tilelang build are skipped.
    """
    names = ["TL_DISABLE_AIU_LOWER"]
    if os.environ.get("TILELANG_TUNE_LDMAT_SWZL", "0") == "1":
        names.append("TL_DISABLE_LDMAT_SWZL")
    dimensions = []
    for name in names:
        key = _pass_config_key(name)
        if key is not None:
            dimensions.append((key, (True, False)))
    return dimensions


def with_aiu_lower_tuning(configs):
    """Expand each config with the PPU pass-config tuning dimensions.

    Always covers TL_DISABLE_AIU_LOWER (True/False); set
    TILELANG_TUNE_LDMAT_SWZL=1 to additionally cover
    TL_DISABLE_LDMAT_SWZL (doubles the space). Only applied on PPU 1.5;
    on PPU 1.0 or non-PPU targets (including tilelang builds without
    these PassConfigKey members) the original configs are returned
    unchanged.
    """
    return _expand_pass_config_tuning(configs, _tuning_dimensions())


# ---------------------------------------------------------------------------
# DeepGEMM FP8 configs
# ---------------------------------------------------------------------------


def get_deepgemm_configs():
    """Tunable configs for FP8 2xAcc GEMM kernel."""
    iter_params = dict(
        block_N=[64, 128],
        num_stages=[2, 3, 4],
        threads=[128, 256],
        enable_rasteration=[True, False],
    )
    return with_aiu_lower_tuning([{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())])


# ---------------------------------------------------------------------------
# MLA decode configs
# ---------------------------------------------------------------------------


def get_mla_configs():
    """Tunable configs for MLA decode kernel."""
    iter_params = dict(
        block_N=[32, 64, 128],
        block_H=[16, 32, 64],
        num_stages=[1, 2, 3],
        threads=[128, 256],
    )
    configs = []
    for values in itertools.product(*iter_params.values()):
        config = {k: v for k, v in zip(iter_params, values)}
        # block_H should not exceed kv_group_num, but we filter at runtime
        configs.append(config)
    return with_aiu_lower_tuning(configs)


# ---------------------------------------------------------------------------
# Paged MLA decode configs
# ---------------------------------------------------------------------------


def get_mla_paged_configs():
    """Tunable configs for paged MLA decode kernel.

    block_N must divide block_size (=64), so options are limited to divisors.
    """
    iter_params = dict(
        block_N=[16, 32, 64],
        block_H=[16, 32, 64],
        num_stages=[1, 2, 3],
        threads=[128, 256],
    )
    return with_aiu_lower_tuning([{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())])


# ---------------------------------------------------------------------------
# NSA (Native Sparse Attention) configs
# ---------------------------------------------------------------------------


def get_nsa_configs():
    """Tunable configs for NSA fwd kernel."""
    iter_params = dict(
        num_stages=[1, 2, 3],
        threads=[32, 64, 128],
    )
    return with_aiu_lower_tuning([{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())])


# ---------------------------------------------------------------------------
# mHC (Multi-Head Contribution) configs
# ---------------------------------------------------------------------------


def get_mhc_pre_configs():
    """Tunable configs for mHC pre kernel (big_fuse + gemm_sqrsum)."""
    iter_params = dict(
        token_block=[16, 32, 64],
        hidden_block=[128, 256, 512],
        num_stages=[1, 2, 3],
    )
    return with_aiu_lower_tuning([{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())])


# ---------------------------------------------------------------------------
# mHC big_fuse configs
# ---------------------------------------------------------------------------


def get_mhc_big_fuse_configs():
    """Tunable configs for mHC pre big_fuse kernel."""
    iter_params = dict(
        threads=[64, 96],
        num_stages=[1, 2, 3],
    )
    return with_aiu_lower_tuning([{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())])


# ---------------------------------------------------------------------------
# mHC post configs
# ---------------------------------------------------------------------------


def get_mhc_post_configs():
    """Tunable configs for mHC post kernel."""
    iter_params = dict(
        n_thr=[64, 128, 256],
        h_blk=[256, 512, 1024],
    )
    return with_aiu_lower_tuning([{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())])


# ---------------------------------------------------------------------------
# NSA decode configs
# ---------------------------------------------------------------------------


def get_nsa_decode_configs():
    """Tunable configs for NSA decode kernel."""
    iter_params = dict(
        num_stages=[0, 1, 2, 3],
        threads=[32, 64],
    )
    return with_aiu_lower_tuning([{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())])


# ---------------------------------------------------------------------------
# V32 sparse MLA fwd configs
# ---------------------------------------------------------------------------


def get_v32_configs():
    """Tunable configs for DeepSeek V32 sparse MLA fwd kernel."""
    iter_params = dict(
        block_I=[64, 128],
        num_stages=[1, 2, 3],
        threads=[128, 256],
    )
    return with_aiu_lower_tuning([{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())])


# ---------------------------------------------------------------------------
# DSA sparse MLA bwd configs
# ---------------------------------------------------------------------------

def get_dsa_bwd_configs():
    """Tunable configs for the DSA sparse MLA bwd kernel.

    block_size must divide topk (512/1024/2048 are all divisible by 32/64)
    and be even because split_store=2 halves each block for the atomic dKV
    store. Larger blocks grow the [padded_H, BS]x[BS, D] gemm tiles and the
    KV shared buffers; configs that exceed shared memory fail to compile and
    are pruned by the autotuner.

    On PPU 1.5 the configs are additionally expanded over the PPU-only pass
    config TL_DISABLE_AIU_LOWER (True/False); set TILELANG_TUNE_LDMAT_SWZL=1
    to also expand over TL_DISABLE_LDMAT_SWZL. On non-PPU1.5 targets (or
    tilelang builds without these PassConfigKey members, e.g. H200) the
    pass-config dimensions are skipped entirely.
    """
    iter_params = dict(
        block_size=[16, 32],
        num_stages=[0, 1, 2],
        threads=[128, 256],
    )
    configs = [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]
    return _expand_pass_config_tuning(configs, _tuning_dimensions())


# ---------------------------------------------------------------------------
# DSA sparse MLA fwd configs
# ---------------------------------------------------------------------------

def get_dsa_fwd_configs():
    """Tunable configs for the DSA sparse MLA fwd kernel.

    block_I must divide topk (512/1024/2048 are all divisible by 16/32/64).
    Larger blocks grow the [H_per_block, BI]x[BI, D] gemm tiles and the KV
    shared buffers; block_I=64 needs 64*576*2B = 72KB of KV shared buffers,
    which exceeds the shared budget on some targets and is pruned by the
    autotuner as a compile failure.

    On PPU 1.5 the configs are additionally expanded over the PPU-only pass
    config TL_DISABLE_AIU_LOWER (True/False); set TILELANG_TUNE_LDMAT_SWZL=1
    to also expand over TL_DISABLE_LDMAT_SWZL. On non-PPU1.5 targets (or
    tilelang builds without these PassConfigKey members, e.g. H200) the
    pass-config dimensions are skipped entirely.
    """
    iter_params = dict(
        block_I=[16, 32, 64],
        num_stages=[1, 2, 3],
        threads=[128, 256],
    )
    configs = [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]
    return _expand_pass_config_tuning(configs, _tuning_dimensions())


# ---------------------------------------------------------------------------
# V32 sparse MLA bwd configs
# ---------------------------------------------------------------------------

def get_v32_bwd_configs():
    """Tunable configs for the DeepSeek V32 sparse MLA bwd kernel.

    Same constraint profile as the DSA bwd: block_size must divide topk
    (512/1024/2048 are all divisible by 32/64) and be even because
    split_store=2 halves each block for the atomic dKV store. The V32 kernel
    additionally vectorizes the atomic dKV store with atomic_addx4, so
    dim/tail_dim must be divisible by 4 (512/64 always are).

    On PPU 1.5 the configs are additionally expanded over the PPU-only pass
    config TL_DISABLE_AIU_LOWER (True/False); set TILELANG_TUNE_LDMAT_SWZL=1
    to also expand over TL_DISABLE_LDMAT_SWZL. On non-PPU1.5 targets (or
    tilelang builds without these PassConfigKey members, e.g. H200) the
    pass-config dimensions are skipped entirely.
    """
    iter_params = dict(
        block_size=[32, 64],
        num_stages=[1, 2],
        threads=[128, 256],
    )
    configs = [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]
    return _expand_pass_config_tuning(configs, _tuning_dimensions())
