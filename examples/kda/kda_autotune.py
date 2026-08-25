import os

from tilelang.autotuner import autotune as _tilelang_autotune


_KDA_DISABLE_AUTOTUNE_ENV = "TILELANG_KDA_DISABLE_AUTOTUNE"


def is_kda_autotune_disabled():
    return os.environ.get(_KDA_DISABLE_AUTOTUNE_ENV, "0").strip().lower() in {"1", "true"}


def autotune(*args, **kwargs):
    if not is_kda_autotune_disabled():
        return _tilelang_autotune(*args, **kwargs)

    def decorator(func):
        return func

    return decorator
