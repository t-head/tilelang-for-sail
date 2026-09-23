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
            new_cfg["pass_configs"] = {tilelang.PassConfigKey.TL_DISABLE_AIU_LOWER: val}
            expanded.append(new_cfg)
    return expanded


class FlashAttentionTuneSpace:
    def __init__(
        self,
        block_sizes_M=(64, 128),
        block_sizes_N=(32, 64),
        thread_options=(128, 256),
        num_stages_range=(1, 2, 3),
    ):
        self.block_sizes_M = block_sizes_M
        self.block_sizes_N = block_sizes_N
        self.thread_options = thread_options
        self.num_stages_range = num_stages_range


def get_configs(user_config=None):
    config = user_config or FlashAttentionTuneSpace()
    valid_configs = []

    # for block_M, block_N in itertools.product(config.block_sizes, repeat=2):
    for block_M in config.block_sizes_M:
        for block_N in config.block_sizes_N:
            for threads in config.thread_options:
                # assert threads % 32 == 0
                # warp_count = threads // 32
                # warp_M = block_M // warp_count
                # warp_N = block_N // warp_count

                if block_M + block_N <= 0.5 * threads:
                    continue

                if block_M + block_N > 2 * threads:
                    continue

                for num_stages in config.num_stages_range:
                    valid_configs.append(
                        {
                            "block_M": block_M,
                            "block_N": block_N,
                            "num_stages": num_stages,
                            "threads": threads,
                        }
                    )
    return with_aiu_lower_tuning(valid_configs)
