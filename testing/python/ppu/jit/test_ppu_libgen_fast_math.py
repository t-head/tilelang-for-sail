import io
import subprocess

import pytest

from tilelang import tvm
from tilelang.jit.adapter import libgen
from tilelang.transform import PassConfigKey


@pytest.mark.parametrize(
    ("pass_configs", "compile_flags", "expected_count"),
    [
        ({}, [], 0),
        ({PassConfigKey.TL_ENABLE_FAST_MATH: False}, [], 0),
        ({PassConfigKey.TL_ENABLE_FAST_MATH: True}, [], 1),
        ({PassConfigKey.TL_ENABLE_FAST_MATH: True}, ["--use_fast_math"], 1),
        ({PassConfigKey.TL_ENABLE_FAST_MATH: False}, ["--use_fast_math"], 1),
    ],
    ids=["default", "disabled", "enabled", "deduplicated", "explicit-compile-flag"],
)
def test_ppu_fast_math_compile_option(monkeypatch, tmp_path, pass_configs, compile_flags, expected_count):
    source_path = tmp_path / "kernel.hg"
    captured_command = []

    class SourceFile(io.StringIO):
        name = str(source_path)

    source_file = SourceFile()
    monkeypatch.setenv("PPU_SDK", str(tmp_path))
    monkeypatch.setattr("tilelang.env.ACTLIZE_INCLUDE_DIR", str(tmp_path / "actlize"))
    monkeypatch.setattr(libgen.tempfile, "NamedTemporaryFile", lambda **kwargs: source_file)

    def fake_run(command, **kwargs):
        captured_command.extend(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(libgen.subprocess, "run", fake_run)

    generator = libgen.LibraryGenerator(tvm.target.Target({"kind": "ppu", "arch": "ppu_15"}))
    generator.assign_pass_configs(pass_configs)
    generator.assign_compile_flags(compile_flags)
    generator.update_lib_code('extern "C" void kernel() {}')
    generator.compile_lib()

    assert captured_command
    assert captured_command.count("--use_fast_math") == expected_count
