"""Tunable parameter configuration generators for DeepSeek kernels.

Each `get_*_configs()` function returns a list of dicts whose keys correspond
to the tunable keyword arguments of the associated kernel function.
"""

import itertools

import tilelang


def with_aiu_lower_tuning(configs):
    """Expand each config with TL_DISABLE_AIU_LOWER True/False variants.

    Only applied on PPU 1.5; on PPU 1.0 or non-PPU targets the original
    configs are returned unchanged.

    PPU 1.5 compute_version is (1, 5).
    """
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
        for val in (True, False):
            new_cfg = dict(cfg)
            new_cfg["pass_configs"] = {
                tilelang.PassConfigKey.TL_DISABLE_AIU_LOWER: val
            }
            expanded.append(new_cfg)
    return expanded


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
    return with_aiu_lower_tuning([
        {k: v for k, v in zip(iter_params, values)}
        for values in itertools.product(*iter_params.values())
    ])


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
    return with_aiu_lower_tuning([
        {k: v for k, v in zip(iter_params, values)}
        for values in itertools.product(*iter_params.values())
    ])


# ---------------------------------------------------------------------------
# NSA (Native Sparse Attention) configs
# ---------------------------------------------------------------------------

def get_nsa_configs():
    """Tunable configs for NSA fwd kernel."""
    iter_params = dict(
        num_stages=[1, 2, 3],
        threads=[32, 64, 128],
    )
    return with_aiu_lower_tuning([
        {k: v for k, v in zip(iter_params, values)}
        for values in itertools.product(*iter_params.values())
    ])


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
    return with_aiu_lower_tuning([
        {k: v for k, v in zip(iter_params, values)}
        for values in itertools.product(*iter_params.values())
    ])


# ---------------------------------------------------------------------------
# mHC big_fuse configs
# ---------------------------------------------------------------------------

def get_mhc_big_fuse_configs():
    """Tunable configs for mHC pre big_fuse kernel."""
    iter_params = dict(
        threads=[64, 96],
        num_stages=[1, 2, 3],
    )
    return with_aiu_lower_tuning([
        {k: v for k, v in zip(iter_params, values)}
        for values in itertools.product(*iter_params.values())
    ])


# ---------------------------------------------------------------------------
# mHC post configs
# ---------------------------------------------------------------------------

def get_mhc_post_configs():
    """Tunable configs for mHC post kernel."""
    iter_params = dict(
        n_thr=[64, 128, 256],
        h_blk=[256, 512, 1024],
    )
    return with_aiu_lower_tuning([
        {k: v for k, v in zip(iter_params, values)}
        for values in itertools.product(*iter_params.values())
    ])


# ---------------------------------------------------------------------------
# NSA decode configs
# ---------------------------------------------------------------------------

def get_nsa_decode_configs():
    """Tunable configs for NSA decode kernel."""
    iter_params = dict(
        num_stages=[0, 1, 2, 3],
        threads=[32, 64],
    )
    return with_aiu_lower_tuning([
        {k: v for k, v in zip(iter_params, values)}
        for values in itertools.product(*iter_params.values())
    ])


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
    return with_aiu_lower_tuning([
        {k: v for k, v in zip(iter_params, values)}
        for values in itertools.product(*iter_params.values())
    ])
