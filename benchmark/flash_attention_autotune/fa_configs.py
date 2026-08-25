import itertools
import argparse

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
                    valid_configs.append({
                        "block_M": block_M,
                        "block_N": block_N,
                        "num_stages": num_stages,
                        "threads": threads,
                    })
    return valid_configs