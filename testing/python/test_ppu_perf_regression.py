#!/usr/bin/env python3
"""Tests for PPU performance regression CI scripts.

Covers:
  - run_current_regression.py: manifest loading, discovery, timeout, payload structure
  - compare_perf_regression.py: comparison logic, fail-closed rules, gate verdicts
  - workflow YAML: structural assertions
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add maint/scripts to path so we can import the modules
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "maint" / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import ppu_run_regression as rcr
import ppu_compare_regression as cpr


# ===========================================================================
# Helpers
# ===========================================================================

def _make_payload(cases: list[dict], schema_version: int = 1, meta: dict | None = None) -> dict:
    """Build a minimal valid payload dict."""
    return {
        "schema_version": schema_version,
        "cases": cases,
        "meta": meta or {},
    }


def _write_payload(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def _make_case(name: str, latency: float | None, status: str = "success",
               file: str = "", error: str = "", source: str = "") -> dict:
    file_identity = file or source or name
    case = {
        "name": name,
        "latency": latency,
        "status": status,
        "file": file_identity,
        "error": error,
        "duration_s": 1.0,
    }
    if source:
        case["source"] = source
    return case


# ===========================================================================
# Tests for run_current_regression.py — manifest loading
# ===========================================================================

class TestManifestLoading:
    """Test manifest parsing and validation."""

    def test_valid_manifest(self, tmp_path: Path):
        """Manifest with valid entries loads correctly."""
        repo = tmp_path / "repo"
        repo.mkdir()
        examples = repo / "examples" / "ppu" / "topk"
        examples.mkdir(parents=True)
        bench = examples / "regression_topk.py"
        bench.write_text("# bench")

        manifest = tmp_path / "manifest.txt"
        manifest.write_text("examples/ppu/topk/regression_topk.py\n")

        result = rcr.load_manifest(str(manifest), repo)
        assert len(result) == 1
        assert result[0] == bench.resolve()

    def test_duplicate_dedup(self, tmp_path: Path):
        """Duplicate entries are removed, order preserved."""
        repo = tmp_path / "repo"
        repo.mkdir()
        d = repo / "examples" / "ppu" / "a"
        d.mkdir(parents=True)
        f = d / "regression_a.py"
        f.write_text("# a")

        manifest = tmp_path / "m.txt"
        manifest.write_text(
            "examples/ppu/a/regression_a.py\n"
            "examples/ppu/a/regression_a.py\n"
        )
        result = rcr.load_manifest(str(manifest), repo)
        assert len(result) == 1

    def test_empty_manifest_fails(self, tmp_path: Path):
        """Empty manifest raises ValueError."""
        repo = tmp_path / "repo"
        repo.mkdir()
        manifest = tmp_path / "empty.txt"
        manifest.write_text("# only comments\n\n")

        with pytest.raises(ValueError, match="empty"):
            rcr.load_manifest(str(manifest), repo)

    def test_path_escape_fails(self, tmp_path: Path):
        """Paths that escape repo root are rejected."""
        repo = tmp_path / "repo"
        repo.mkdir()
        manifest = tmp_path / "m.txt"
        manifest.write_text("../../etc/regression_evil.py\n")

        with pytest.raises(ValueError, match="escapes repo root"):
            rcr.load_manifest(str(manifest), repo)

    def test_missing_file_fails(self, tmp_path: Path):
        """Non-existent file in manifest raises FileNotFoundError."""
        repo = tmp_path / "repo"
        repo.mkdir()
        manifest = tmp_path / "m.txt"
        manifest.write_text("examples/ppu/missing/regression_missing.py\n")

        with pytest.raises(FileNotFoundError, match="file not found"):
            rcr.load_manifest(str(manifest), repo)

    def test_non_regression_file_fails(self, tmp_path: Path):
        """Files not matching regression_*.py are rejected."""
        repo = tmp_path / "repo"
        repo.mkdir()
        d = repo / "examples" / "ppu"
        d.mkdir(parents=True)
        f = d / "benchmark_foo.py"
        f.write_text("# bench")

        manifest = tmp_path / "m.txt"
        manifest.write_text("examples/ppu/benchmark_foo.py\n")

        with pytest.raises(ValueError, match="regression_"):
            rcr.load_manifest(str(manifest), repo)

    def test_manifest_file_not_found(self, tmp_path: Path):
        """Missing manifest file raises FileNotFoundError."""
        repo = tmp_path / "repo"
        repo.mkdir()
        with pytest.raises(FileNotFoundError, match="Manifest file not found"):
            rcr.load_manifest(str(tmp_path / "no_such.txt"), repo)


class TestDiscovery:
    """Test default auto-discovery (no manifest)."""

    def test_discover_finds_regression_files(self, tmp_path: Path):
        """Auto-discovery finds regression_*.py files."""
        examples = tmp_path / "examples"
        (examples / "ppu" / "a").mkdir(parents=True)
        (examples / "ppu" / "a" / "regression_a.py").write_text("# a")
        (examples / "ppu" / "a" / "other.py").write_text("# other")

        result = rcr.discover_bench_files(examples)
        assert len(result) == 1
        assert result[0].name == "regression_a.py"


# ===========================================================================
# Tests for run_current_regression.py — payload structure
# ===========================================================================

class TestPayloadStructure:
    """Test payload building."""

    def test_payload_schema_version(self):
        """Payload has schema_version=1."""
        results = [rcr.CaseResult("t", 1.5, "success", "f.py")]
        payload = rcr.build_payload(results)
        assert payload.schema_version == 1

    def test_nan_latency_becomes_none(self):
        """NaN latency is serialized as None."""
        results = [rcr.CaseResult("t", float("nan"), "failed", "f.py", error="boom")]
        payload = rcr.build_payload(results)
        d = payload.to_dict()
        assert d["cases"][0]["latency"] is None

    def test_timeout_in_payload(self):
        """Timeout status preserved in payload."""
        results = [rcr.CaseResult("t", float("nan"), "timeout", "f.py", error="timed out")]
        payload = rcr.build_payload(results)
        d = payload.to_dict()
        assert d["cases"][0]["status"] == "timeout"


class TestParseOutput:
    """Test output parsing."""

    def test_json_prefix_parsing(self):
        """JSON prefix line is parsed correctly."""
        output = 'some log\n__TILELANG_PERF_RESULTS_JSON__={"a":1.5,"b":2.0}\n'
        result = rcr.parse_output(output)
        assert result == {"a": 1.5, "b": 2.0}

    def test_json_list_format(self):
        """JSON list format is parsed correctly."""
        data = [{"name": "x", "latency": 3.0}]
        output = f'__TILELANG_PERF_RESULTS_JSON__={json.dumps(data)}\n'
        result = rcr.parse_output(output)
        assert result == {"x": 3.0}

    def test_text_fallback(self):
        """Text key:value fallback works."""
        output = "bench_a: 1.5\nbench_b: 2.0\n"
        result = rcr.parse_output(output)
        assert result == {"bench_a": 1.5, "bench_b": 2.0}

    def test_no_marker_returns_none_from_json_parser(self):
        """_parse_json_output returns None when no marker is present."""
        assert rcr._parse_json_output("some noise\nno marker here\n") is None

    def test_empty_list_marker_returns_empty_dict(self):
        """Empty list JSON marker returns {} (not None), suppressing text fallback."""
        output = "__TILELANG_PERF_RESULTS_JSON__=[]\n"
        assert rcr._parse_json_output(output) == {}

    def test_empty_dict_marker_returns_empty_dict(self):
        """Empty dict JSON marker returns {} (not None), suppressing text fallback."""
        output = "__TILELANG_PERF_RESULTS_JSON__={}\n"
        assert rcr._parse_json_output(output) == {}

    def test_empty_list_marker_with_capability_noise_no_fake_latency(self):
        """Empty list marker + CUDA capability noise must NOT produce fake latency.

        This is the core regression scenario: the benchmark emits an empty
        JSON result followed by device-probe lines like
        ``CUDA device capability: 89``.  The old truthiness check caused
        fallback to text parsing which picked up 89 as a latency value.
        """
        output = (
            "Loading PPU driver...\n"
            "__TILELANG_PERF_RESULTS_JSON__=[]\n"
            "CUDA device capability: 89\n"
            "Device 0: Some PPU Card\n"
        )
        result = rcr.parse_output(output)
        assert result == {}
        # Crucially, 89 must NOT appear as a latency value
        assert 89.0 not in result.values()

    def test_empty_dict_marker_with_capability_noise_no_fake_latency(self):
        """Empty dict marker + noise lines also returns empty dict."""
        output = (
            "__TILELANG_PERF_RESULTS_JSON__={}\n"
            "CUDA device capability: 89\n"
        )
        result = rcr.parse_output(output)
        assert result == {}


class TestRunSingleBenchmark:
    """Exercise the real subprocess path, including pipe and timeout behavior."""

    @staticmethod
    def _script(tmp_path: Path, name: str, body: str) -> tuple[Path, Path]:
        repo = tmp_path / "repo"
        bench_dir = repo / "examples" / "ppu" / "test"
        bench_dir.mkdir(parents=True, exist_ok=True)
        script = bench_dir / name
        script.write_text(body)
        return repo, script

    def test_success_json_subprocess(self, tmp_path: Path):
        repo, script = self._script(
            tmp_path,
            "regression_success.py",
            'print(\'__TILELANG_PERF_RESULTS_JSON__={"case_a":1.25}\')\n',
        )
        results = rcr.run_single_benchmark(script, repo, timeout=2)
        assert [(r.name, r.status, r.latency) for r in results] == [("case_a", "success", 1.25)]
        assert results[0].source == "examples/ppu/test/regression_success.py"

    def test_nonzero_subprocess_preserves_traceback(self, tmp_path: Path):
        repo, script = self._script(
            tmp_path,
            "regression_failure.py",
            "import sys\nprint('before failure')\nprint('traceback marker', file=sys.stderr)\nraise SystemExit(3)\n",
        )
        result = rcr.run_single_benchmark(script, repo, timeout=2)[0]
        assert result.status == "failed"
        assert "Exit code 3" in result.error
        assert "before failure" in result.error
        assert "traceback marker" in result.error

    def test_large_stderr_does_not_deadlock(self, tmp_path: Path):
        repo, script = self._script(
            tmp_path,
            "regression_stderr.py",
            "import sys\nsys.stderr.write('x' * 200000)\n"
            "print('__TILELANG_PERF_RESULTS_JSON__={\"case_a\":2.5}')\n",
        )
        result = rcr.run_single_benchmark(script, repo, timeout=3)[0]
        assert result.status == "success"
        assert result.latency == 2.5

    def test_sleep_subprocess_times_out(self, tmp_path: Path):
        repo, script = self._script(
            tmp_path,
            "regression_timeout.py",
            "import time\nprint('started', flush=True)\ntime.sleep(5)\n",
        )
        result = rcr.run_single_benchmark(script, repo, timeout=1)[0]
        assert result.status == "timeout"
        assert "Timed out after 1s" in result.error
        assert result.duration_s < 4

    def test_missing_subprocess_is_not_found(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        missing = repo / "examples" / "regression_missing.py"
        result = rcr.run_single_benchmark(missing, repo, timeout=1)[0]
        assert result.status == "not_found"
        assert result.source == "examples/regression_missing.py"

    def test_partial_manifest_writes_payload_and_exits_nonzero(self, tmp_path: Path):
        import subprocess

        repo = tmp_path / "repo"
        bench_dir = repo / "examples" / "ppu"
        bench_dir.mkdir(parents=True)
        good = bench_dir / "regression_good.py"
        bad = bench_dir / "regression_bad.py"
        good.write_text('print(\'__TILELANG_PERF_RESULTS_JSON__={"good":1.0}\')\n')
        bad.write_text("raise RuntimeError('broken benchmark')\n")
        manifest = repo / "manifest.txt"
        manifest.write_text("examples/ppu/regression_good.py\nexamples/ppu/regression_bad.py\n")
        output = repo / "payload.json"

        completed = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "ppu_run_regression.py"),
                "--manifest", str(manifest),
                "--repo-root", str(repo),
                "--timeout", "2",
                "--output", str(output),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 1
        payload = json.loads(output.read_text())
        assert {case["status"] for case in payload["cases"]} == {"success", "failed"}
        assert all(case["source"].startswith("examples/ppu/") for case in payload["cases"])

    def test_relative_repo_root_cli(self, tmp_path: Path):
        """CLI with --repo-root as relative path must not raise ValueError.

        Regression test: load_manifest resolves paths to absolute while a
        relative repo_root caused relative_to() to fail inside main().
        """
        import os
        import subprocess as sp

        repo = tmp_path / "repo"
        bench_dir = repo / "examples" / "ppu"
        bench_dir.mkdir(parents=True)
        bench = bench_dir / "regression_rel.py"
        bench.write_text('print(\'__TILELANG_PERF_RESULTS_JSON__={"k":1.0}\')\n')
        manifest = repo / "manifest.txt"
        manifest.write_text("examples/ppu/regression_rel.py\n")
        output = repo / "payload.json"

        completed = sp.run(
            [
                sys.executable,
                str(_SCRIPTS_DIR / "ppu_run_regression.py"),
                "--manifest", str(manifest),
                "--repo-root", ".",
                "--timeout", "5",
                "--output", str(output),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(repo),
        )
        assert completed.returncode == 0, (
            f"Expected exit 0, got {completed.returncode}\n"
            f"STDERR: {completed.stderr[-2000:]}"
        )
        payload = json.loads(output.read_text())
        assert payload["cases"][0]["status"] == "success"

    def test_empty_json_with_noise_no_fake_case(self, tmp_path: Path):
        """A benchmark emitting empty JSON marker + CUDA capability noise
        must NOT fabricate a case with latency 89.

        Empty JSON marker is a legitimate result (zero cases), not a parse
        failure.  The subprocess path must never fall through to text
        parsing which would pick up '89' from capability noise.
        """
        repo, script = self._script(
            tmp_path,
            "regression_empty.py",
            (
                "import sys\n"
                "print('Loading PPU driver...')\n"
                "print('__TILELANG_PERF_RESULTS_JSON__=[]')\n"
                "print('CUDA device capability: 89')\n"
            ),
        )
        results = rcr.run_single_benchmark(script, repo, timeout=5)
        # Empty JSON marker → zero cases, NOT a "failed" status
        assert results == []
        # Must not contain 89 anywhere as latency
        for r in results:
            if r.status == "success":
                assert r.latency != 89.0, "Fabricated case from CUDA capability noise"

    def test_empty_dict_marker_subprocess_returns_no_cases(self, tmp_path: Path):
        """Empty dict JSON marker via subprocess returns zero cases (not failure)."""
        repo, script = self._script(
            tmp_path,
            "regression_empty_dict.py",
            "print('__TILELANG_PERF_RESULTS_JSON__={}')\n",
        )
        results = rcr.run_single_benchmark(script, repo, timeout=5)
        assert results == []
        assert all(r.latency != 89.0 for r in results if r.status == "success")

    def test_relative_repo_root_subprocess(self, tmp_path: Path):
        """run_single_benchmark works when repo_root is a relative Path.

        Regression test for the ValueError when bench_file.relative_to()
        was called on an un-resolved relative repo_root.
        """
        repo, script = self._script(
            tmp_path,
            "regression_relroot.py",
            'print(\'__TILELANG_PERF_RESULTS_JSON__={"case_rel":0.5}\')\n',
        )
        # Build a relative path to the repo directory
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            rel_repo = Path(os.path.relpath(repo, tmp_path))
            results = rcr.run_single_benchmark(script, rel_repo, timeout=5)
            assert len(results) == 1
            assert results[0].name == "case_rel"
            assert results[0].status == "success"
        finally:
            os.chdir(old_cwd)


# ===========================================================================
# Tests for compare_perf_regression.py
# ===========================================================================

class TestCompare:
    """Test comparison logic and fail-closed rules."""

    def test_improvement(self):
        """Speedup > 1 classified as improved."""
        old = _make_payload([_make_case("a", 2.0)])
        new = _make_payload([_make_case("a", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert result["comparisons"][0]["status"] == "improved"
        assert result["comparisons"][0]["speedup"] == 2.0
        assert result["gate"]["correctness_ok"] is True

    def test_single_regression(self):
        """Speedup < threshold classified as regression."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", 2.0)])  # new is slower
        result = cpr.compare_payloads(old, new, regression_threshold=0.95)
        assert result["comparisons"][0]["status"] == "regressed"
        assert result["gate"]["has_performance_regression"] is True
        # Phase 0: correctness is still OK
        assert result["gate"]["correctness_ok"] is True

    def test_two_regressions(self):
        """Multiple regressions are counted."""
        old = _make_payload([_make_case("a", 1.0), _make_case("b", 1.0)])
        new = _make_payload([_make_case("a", 2.0), _make_case("b", 3.0)])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["num_regressions"] == 2

    def test_zero_common_success(self):
        """Zero common success cases -> correctness failure."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("b", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False
        assert "zero_common_success" in result["gate"]["correctness_failures"]

    def test_old_success_new_failed(self):
        """Old success but new failed -> correctness failure."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", None, status="failed", error="crash")])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False
        assert "old_success_new_failed" in result["gate"]["correctness_failures"]

    def test_old_success_new_timeout(self):
        """Old success but new timeout -> correctness failure."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", None, status="timeout")])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False
        assert "old_success_new_failed" in result["gate"]["correctness_failures"]

    def test_all_old_failed(self):
        """All old cases failed -> correctness failure."""
        old = _make_payload([_make_case("a", None, status="failed")])
        new = _make_payload([_make_case("a", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False
        assert "all_old_failed" in result["gate"]["correctness_failures"]

    def test_all_new_failed(self):
        """All new cases failed -> correctness failure."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", None, status="failed")])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False

    def test_only_old_only_new_reported(self):
        """Only-old and only-new cases are reported but don't fail."""
        old = _make_payload([_make_case("shared", 1.0), _make_case("only_old", 1.0)])
        new = _make_payload([_make_case("shared", 1.0), _make_case("only_new", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert result["only_old"] == ["only_old"]
        assert result["only_new"] == ["only_new"]
        assert result["gate"]["correctness_ok"] is True

    def test_invalid_latency_zero(self):
        """Latency of 0 is invalid."""
        old = _make_payload([_make_case("a", 0.0)])
        new = _make_payload([_make_case("a", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert "invalid_latency" in result["gate"]["correctness_failures"]

    def test_invalid_latency_negative(self):
        """Negative latency is invalid."""
        old = _make_payload([_make_case("a", -1.0)])
        new = _make_payload([_make_case("a", 1.0)])
        result = cpr.compare_payloads(old, new)
        assert "invalid_latency" in result["gate"]["correctness_failures"]

    def test_multicase_source_failure_marks_all_old_cases(self):
        source = "examples/ppu/topk/regression_topk.py"
        old = _make_payload([
            _make_case("topk_a", 1.0, source=source),
            _make_case("topk_b", 2.0, source=source),
        ])
        new = _make_payload([
            _make_case(source, None, status="timeout", source=source, error="timeout"),
        ])
        result = cpr.compare_payloads(old, new)
        assert result["gate"]["correctness_ok"] is False
        assert "source_failure" in result["gate"]["correctness_failures"]
        assert result["source_failures"][0]["affected_old_cases"] == ["topk_a", "topk_b"]

    def test_source_failure_blocks_when_other_source_succeeds(self):
        failed_source = "examples/ppu/topk/regression_topk.py"
        good_source = "examples/ppu/flash/regression_flash.py"
        old = _make_payload([
            _make_case("topk_a", 1.0, source=failed_source),
            _make_case("flash_a", 2.0, source=good_source),
        ])
        new = _make_payload([
            _make_case(failed_source, None, status="failed", source=failed_source),
            _make_case("flash_a", 1.9, source=good_source),
        ])
        result = cpr.compare_payloads(old, new)
        assert result["summary"]["common_success"] == 1
        assert result["gate"]["correctness_ok"] is False
        assert result["gate"]["source_failures"][0]["source"] == failed_source
        assert "Source-level Failures" in cpr.generate_markdown(result)

    def test_schema_error(self, tmp_path: Path):
        """Wrong schema version raises ValueError."""
        bad = _write_payload(tmp_path, "bad.json", {"schema_version": 99, "cases": []})
        with pytest.raises(ValueError, match="schema_version"):
            cpr.load_payload(str(bad))


class TestCompareMarkdown:
    """Test Markdown report generation."""

    def test_markdown_contains_verdict(self):
        """Report contains a verdict line."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", 1.0)])
        result = cpr.compare_payloads(old, new)
        md = cpr.generate_markdown(result)
        assert "**Verdict:" in md

    def test_markdown_fail_verdict(self):
        """Correctness failure shows FAIL in verdict."""
        old = _make_payload([_make_case("a", 1.0)])
        new = _make_payload([_make_case("a", None, status="failed")])
        result = cpr.compare_payloads(old, new)
        md = cpr.generate_markdown(result)
        assert "FAIL" in md



# ===========================================================================
# Tests for CLI help (smoke tests)
# ===========================================================================

class TestCLIHelp:
    """Verify CLI --help works without error."""

    def test_run_current_regression_help(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, str(_SCRIPTS_DIR / "ppu_run_regression.py"), "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "--manifest" in r.stdout
        assert "--timeout" in r.stdout

    def test_compare_perf_regression_help(self):
        import subprocess
        r = subprocess.run(
            [sys.executable, str(_SCRIPTS_DIR / "ppu_compare_regression.py"), "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "--old" in r.stdout
        assert "--new" in r.stdout



# ===========================================================================
# Tests for workflow YAML structure
# ===========================================================================

class TestWorkflowYAML:
    """Lightweight structural tests on the workflow YAML."""

    @pytest.fixture(autouse=True)
    def load_workflow(self):
        import yaml
        wf_path = _REPO_ROOT / ".github" / "workflows" / "ppu-perf-regression-bot.yml"
        with open(wf_path) as f:
            self.wf = yaml.safe_load(f)

    def test_triggers(self):
        """Workflow is driven exclusively by PR issue_comment events."""
        # PyYAML parses bare 'on:' as boolean True key
        on = self.wf.get("on") or self.wf.get(True)
        assert on is not None, "No 'on' trigger block found"
        assert set(on.keys()) == {"issue_comment"}
        assert on["issue_comment"] == {"types": ["created"]}
        assert "workflow_dispatch" not in on
        assert "push" not in on
        assert "pull_request" not in on

    def test_permissions(self):
        """Workflow has minimal required permissions."""
        perms = self.wf["permissions"]
        assert perms["contents"] == "read"
        assert perms["issues"] == "write"
        assert perms["pull-requests"] == "write"

    def test_concurrency(self):
        """Workflow has PR-based concurrency group that only reads issue.number."""
        conc = self.wf["concurrency"]
        group = conc["group"]
        assert "github.event.issue.number" in group
        assert "inputs" not in group
        assert "workflow_dispatch" not in group
        assert conc["cancel-in-progress"] is True

    def test_resolve_pr_job_exists(self):
        """resolve-pr job exists with outputs."""
        jobs = self.wf["jobs"]
        assert "resolve-pr" in jobs
        assert "outputs" in jobs["resolve-pr"]
        outputs = jobs["resolve-pr"]["outputs"]
        for key in ["pr_number", "base_ref", "base_sha", "head_sha"]:
            assert key in outputs, f"Missing output: {key}"

    def test_pre_commit_lint_job_configuration(self):
        """Quick Lint is a strict downstream gate of resolve-pr."""
        job = self.wf["jobs"]["pre-commit-lint"]
        assert job["name"] == "Quick Lint"
        assert job["needs"] == "resolve-pr"
        assert job["runs-on"] == "ubuntu-latest"
        assert job["timeout-minutes"] == 30
        assert "continue-on-error" not in job
        assert "always()" not in str(job)

    def test_pre_commit_lint_checkout_resolved_merge_ref(self):
        """Lint checks out the resolved merge ref with complete submodules."""
        steps = self.wf["jobs"]["pre-commit-lint"]["steps"]
        checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
        assert checkout["uses"] == "actions/checkout@v7"
        assert checkout["with"]["ref"] == "${{ needs.resolve-pr.outputs.merge_ref }}"
        assert checkout["with"]["fetch-depth"] == 0
        assert checkout["with"]["submodules"] == "recursive"

    def test_pre_commit_lint_python_and_commands(self):
        """Lint uses Python 3.10 and the exact baseline commands."""
        steps = self.wf["jobs"]["pre-commit-lint"]["steps"]
        setup = next(step for step in steps if step.get("uses", "").startswith("actions/setup-python@"))
        assert setup["uses"] == "actions/setup-python@v6"
        assert setup["with"]["python-version"] == "3.10"
        commands = [step.get("run") for step in steps if "run" in step]
        assert commands == [
            "python -m compileall -q -f tilelang",
            "pipx run pre-commit run --all-files --color=always --show-diff-on-failure",
        ]
        assert all("continue-on-error" not in step for step in steps)
        assert all("always()" not in str(step.get("if", "")) for step in steps)

    def test_ppu_perf_job_exists(self):
        """ppu-perf waits for both PR resolution and the lint gate."""
        jobs = self.wf["jobs"]
        assert "ppu-perf" in jobs
        job = jobs["ppu-perf"]
        assert job["needs"] == ["resolve-pr", "pre-commit-lint"]
        assert "if" not in job

    def test_scheduler_step_exists_only_in_ppu_perf(self):
        """Only the gated PPU job can start the hardware scheduler action."""
        scheduler_jobs = []
        for job_name, job in self.wf["jobs"].items():
            if any("ppu-scheduler-action" in step.get("uses", "") for step in job.get("steps", [])):
                scheduler_jobs.append(job_name)
        assert scheduler_jobs == ["ppu-perf"]

    def test_ppu_perf_timeout(self):
        """PPU job has 360-minute timeout."""
        job = self.wf["jobs"]["ppu-perf"]
        assert job["timeout-minutes"] == 360

    def test_ppu_scheduler_action(self):
        """PPU job uses ppu-scheduler-action with correct board type."""
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        bench_step = None
        for s in steps:
            if s.get("uses", "").startswith("t-head/ppu-scheduler-action"):
                bench_step = s
                break
        assert bench_step is not None, "ppu-scheduler-action step not found"
        assert "ZW-M890P" in bench_step["with"]["node_selector"]
        assert bench_step["with"]["timeout_minutes"] == 300

    def test_no_hardcoded_main(self):
        """Pod command does not hardcode 'git checkout main'."""
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        for s in steps:
            if "uses" in s and "ppu-scheduler-action" in s["uses"]:
                cmd = s["with"].get("command", "")
                assert "git checkout main" not in cmd

    def test_owner_check_t_head(self):
        """Workflow checks for t-head owner, not tile-ai."""
        # Check in the resolve-pr job steps
        steps = self.wf["jobs"]["resolve-pr"]["steps"]
        found = False
        for s in steps:
            script = s.get("with", {}).get("script", "")
            if "t-head" in script:
                found = True
                break
        assert found, "t-head owner check not found in resolve-pr"

    def test_issue_comment_filter(self):
        """resolve-pr gate accepts only the exact @ppu-regression comment on PRs."""
        resolve_job = self.wf["jobs"]["resolve-pr"]
        if_cond = resolve_job.get("if", "")
        assert "github.event_name == 'issue_comment'" in if_cond
        assert "github.event.issue.pull_request" in if_cond
        assert "github.event.comment.body == '@ppu-regression'" in if_cond
        assert "workflow_dispatch" not in if_cond
        assert "contains(" not in if_cond

    def test_permission_check_reads_only_comment_user(self):
        """Permission check uses only comment author; no context.actor fallback."""
        steps = self.wf["jobs"]["resolve-pr"]["steps"]
        permission_step = next(step for step in steps if step["name"] == "Check permissions")
        assert "if" not in permission_step
        script = permission_step["with"]["script"]
        assert "context.payload.comment?.user?.login" in script
        assert "context.actor" not in script
        assert "context.eventName" not in script
        assert "['admin', 'maintain', 'write']" in script

    def test_pr_number_sourced_only_from_issue_number(self):
        """PR number is read strictly from issue payload, never dispatch inputs."""
        steps = self.wf["jobs"]["resolve-pr"]["steps"]
        resolve_step = next(step for step in steps if step["name"] == "Resolve PR details")
        script = resolve_step["with"]["script"]
        assert "context.payload.issue?.number" in script
        assert "context.payload.inputs" not in script
        assert "workflow_dispatch" not in script
        assert "/^[1-9][0-9]*$/" in script
        assert "${{" not in script

    def test_start_comment_posts_without_dispatch_guard(self):
        """Start comment upsert is unconditional on the comment-only workflow."""
        steps = self.wf["jobs"]["resolve-pr"]["steps"]
        start_step = next(step for step in steps if step["name"] == "Post start comment")
        assert "workflow_dispatch" not in str(start_step.get("if", ""))
        script = start_step["with"]["script"]
        assert "updateComment" in script
        assert "createComment" in script

    def test_both_venvs_inherit_system_site_packages(self):
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        scheduler = next(step for step in steps if "ppu-scheduler-action" in step.get("uses", ""))
        command = scheduler["with"]["command"]
        assert command.count("python -m venv --system-site-packages") == 2
        assert '"$WORKSPACE/venv_new"' in command
        assert '"$WORKSPACE/venv_old"' in command

    def test_scheduler_extra_env_is_single_line_without_spaces(self):
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        scheduler = next(step for step in steps if "ppu-scheduler-action" in step.get("uses", ""))
        extra_env = scheduler["with"]["extra_env"]
        assert "\n" not in extra_env
        assert ", " not in extra_env
        assert extra_env.count(",") == 5

    def test_missing_gate_is_fail_closed_in_comment_and_final_verdict(self):
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        comment = next(step for step in steps if step["name"] == "Post results to PR")
        comment_script = comment["with"]["script"]
        assert "gateJson.correctness_ok === true" in comment_script
        assert "Infrastructure failure" in comment_script
        verdict = next(step for step in steps if step["name"] == "Gate verdict")
        assert "Gate JSON is missing, invalid, or reports a correctness failure" in verdict["run"]

    def test_runner_exit_codes_are_captured_not_discarded(self):
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        scheduler = next(step for step in steps if "ppu-scheduler-action" in step.get("uses", ""))
        command = scheduler["with"]["command"]
        assert "|| OLD_RC=$?" in command
        assert "|| NEW_RC=$?" in command
        assert "old_payload.json\" || true" not in command
        assert "new_payload.json\" || true" not in command

    def test_ppu_image_is_torch_210(self):
        """PPU_IMAGE must reference the CUPTI-verified torch 2.10 image."""
        expected = (
            "download.t-head.cn/docker_release/pytorch:"
            "sdk2.2.0-2.10.0-ubuntu24.04-hggcrt3-py312-20261005"
        )
        assert self.wf.get("env", {}).get("PPU_IMAGE") == expected

    def test_scheduler_image_uses_ppu_image_env(self):
        """The scheduler image must use the workflow PPU_IMAGE variable."""
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        scheduler = next(
            step for step in steps
            if "ppu-scheduler-action" in step.get("uses", "")
        )
        assert scheduler["with"]["image"] == "${{ env.PPU_IMAGE }}"

    def test_workflow_uses_tracked_txt_manifest(self):
        """The runner command and PR comment must reference the TXT manifest."""
        steps = self.wf["jobs"]["ppu-perf"]["steps"]
        scheduler = next(
            step for step in steps
            if "ppu-scheduler-action" in step.get("uses", "")
        )
        command = scheduler["with"]["command"]
        assert 'MANIFEST="$NEW_SRC/maint/scripts/ppu_perf_890p_allowlist.txt"' in command
        assert "ppu_perf_890p_allowlist.yaml" not in command

        comment = next(step for step in steps if step["name"] == "Post results to PR")
        script = comment["with"]["script"]
        assert "maint/scripts/ppu_perf_890p_allowlist.txt" in script
        assert "ppu_perf_890p_allowlist.yaml" not in script


# ===========================================================================
# Tests for the tracked repository ci.yml (functional CI)
# ===========================================================================

class TestFunctionalCIYAML:
    """Structural assertions on the tracked functional CI workflow."""

    @pytest.fixture(autouse=True)
    def load_ci(self):
        import yaml
        wf_path = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
        with open(wf_path) as f:
            self.ci = yaml.safe_load(f)

    def test_ppu_tests_depends_on_lint(self):
        """ppu-tests is gated on the Quick Lint job via the needs default propagation."""
        jobs = self.ci["jobs"]
        assert "ppu-tests" in jobs
        assert "lint" in jobs
        needs = jobs["ppu-tests"]["needs"]
        if isinstance(needs, str):
            needs = [needs]
        assert "lint" in needs

    def test_ppu_tests_has_no_bypass_conditions(self):
        """ppu-tests must not bypass needs failure propagation."""
        job = self.ci["jobs"]["ppu-tests"]
        # No job-level if that would unlock the job when lint fails/cancels
        job_if = job.get("if", "")
        assert "always()" not in str(job_if)
        assert "!cancelled()" not in str(job_if)
        assert "cancelled() == false" not in str(job_if)
        assert "continue-on-error" not in job

    def test_ppu_tests_matrix_defaults_two_boards(self):
        """The matrix fallback still provides both ZW-M890P and OAM-810E defaults."""
        job = self.ci["jobs"]["ppu-tests"]
        board_expr = job["strategy"]["matrix"]["board"]
        assert "ZW-M890P" in board_expr
        assert "OAM-810E" in board_expr
        assert "inputs.board_matrix" in board_expr

    def test_scheduler_step_only_in_ppu_tests(self):
        """ppu-scheduler-action is still only reachable through ppu-tests."""
        scheduler_jobs = []
        for job_name, job in self.ci["jobs"].items():
            steps = job.get("steps", []) or []
            if any("ppu-scheduler-action" in step.get("uses", "") for step in steps):
                scheduler_jobs.append(job_name)
        assert scheduler_jobs == ["ppu-tests"]

    def test_lint_job_contract_preserved(self):
        """The baseline lint job still carries the exact pre-commit contract."""
        lint = self.ci["jobs"]["lint"]
        assert lint["runs-on"] == "ubuntu-latest"
        assert lint["timeout-minutes"] == 30
        runs = [step.get("run", "") for step in lint["steps"] if "run" in step]
        assert any("-m compileall -q -f tilelang" in r for r in runs)
        assert any("pipx run pre-commit run --all-files --color=always --show-diff-on-failure" in r for r in runs)

    def test_on_block_unchanged_from_baseline(self):
        """We did not alter the baseline trigger surface of ci.yml."""
        on = self.ci.get("on") or self.ci.get(True)
        assert on is not None
        assert "push" in on
        assert "pull_request" in on
        assert "workflow_dispatch" in on


# ===========================================================================
# Tests for the shipped PPU 890P allowlist manifest
# ===========================================================================

_MANIFEST_PATH = _SCRIPTS_DIR / "ppu_perf_890p_allowlist.txt"

_EXPECTED_PPU_ENTRIES = (
    "examples/ppu/flash_decoding/regression_example_flash_decoding.py",
    "examples/ppu/seer_attention/regression_block_sparse_attn_tilelang.py",
    "examples/ppu/topk/regression_topk_tilelang.py",
)

_EXPECTED_NEW_ENTRIES = (
    # attention
    "examples/attention_sink/regression_attention_sink.py",
    "examples/blocksparse_attention/regression_example_blocksparse_attention.py",
    "examples/flash_attention/regression_example_flash_attention.py",
    "examples/flash_decoding/regression_example_flash_decoding.py",
    "examples/seer_attention/regression_block_sparse_attn_tilelang.py",
    # gemm
    "examples/gemm_fp8/regression_example_gemm_fp8.py",
    "examples/gemm_splitk/regression_example_gemm_splitk.py",
    "examples/blocksparse_gemm/regression_example_blocksparse_gemm.py",
    # moe
    "examples/fusedmoe/regression_example_fusedmoe.py",
    # deepseek
    "examples/deepseek_mhc/regression_example_mhc.py",
    "examples/deepseek_nsa/regression_example_tilelang_nsa.py",
)


class TestShippedManifest:
    """Structural assertions on the tracked PPU 890P allowlist manifest."""

    @pytest.fixture(autouse=True)
    def load_entries(self):
        # Delegate to the runner's loader to avoid duplicating validation.
        resolved = rcr.load_manifest(str(_MANIFEST_PATH), _REPO_ROOT)
        self.resolved: list[Path] = resolved
        self.relative: list[str] = [
            str(p.resolve().relative_to(_REPO_ROOT.resolve())) for p in resolved
        ]

    def test_total_valid_entry_count(self):
        """Manifest resolves to exactly 14 verified entries."""
        assert len(self.resolved) == 14

    def test_no_duplicate_paths(self):
        """Each repo-relative entry is unique."""
        assert len(set(self.relative)) == len(self.relative)

    def test_original_ppu_entries_preserved(self):
        """The three original PPU-specific entries are still present."""
        for entry in _EXPECTED_PPU_ENTRIES:
            assert entry in self.relative, f"Missing original PPU entry: {entry}"

    def test_new_target_entries_present(self):
        """All 11 verified attention/gemm/moe/deepseek entries are present."""
        for entry in _EXPECTED_NEW_ENTRIES:
            assert entry in self.relative, f"Missing target entry: {entry}"

    def test_entries_are_existing_regression_files(self):
        """Every resolved path exists on disk and matches regression_*.py."""
        for p in self.resolved:
            assert p.is_file(), f"Not a file: {p}"
            assert p.name.startswith("regression_") and p.name.endswith(".py"), (
                f"basename does not match regression_*.py: {p.name}"
            )

    def test_entries_match_expected_union(self):
        """Resolved entries equal the union of original + target entries (set equality)."""
        expected = set(_EXPECTED_PPU_ENTRIES) | set(_EXPECTED_NEW_ENTRIES)
        assert set(self.relative) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
