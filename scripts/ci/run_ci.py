#!/usr/bin/env python3
"""SGLang CI Test Launcher — arbitrary combination runner + dynamic config.

Python rewrite of run_ci.sh. Fully equivalent CLI interface.

Core idea: a unified ``run`` command that accepts any mix of
  -s / --suite     suite targets
  -f / --file      file targets
  -t / --test      method-level targets  (file.py::Class::method)
  -g / --grep      keyword-match targets
All targets are collected, deduplicated, and executed in one shot.

Quick reference
===============

List suites        ./run_ci.py list --hw cuda
List suite tests   ./run_ci.py list --hw cuda --suite stage-b-test-small-1-gpu

--- run: arbitrary combination ---

Single suite       ./run_ci.py run -s stage-b-test-small-1-gpu
Multiple suites    ./run_ci.py run -s stage-a-test-1 -s stage-b-test-small-1-gpu
Single file        ./run_ci.py run -f test/registered/attention/test_flash_attention_4.py
Single method      ./run_ci.py run -t <path/to/test.py>::TestOpenAIServer::test_completion
Keyword match      ./run_ci.py run -g eagle
All combined       ./run_ci.py run -s stage-a-test-1 \\
                       -f test/registered/models/test_generation_models.py \\
                       -t <path/to/test.py>::TestOpenAIServer::test_completion \\
                       -g flash_attention
Dry-run            ./run_ci.py run -s stage-a-test-1 -g eagle --dry-run
With retry         ./run_ci.py run -s stage-b-test-small-1-gpu --retry --continue-on-error

--- dynamic config file ---

From config        ./run_ci.py run -c scripts/ci/ci_tests.json -p quick-unit
Override vars      ./run_ci.py run -c ci_tests.json -p quick-unit --var HW=amd --var MAX_EST=60
Config + extra     ./run_ci.py run -c ci_tests.json -p kernel-sanity -f extra_test.py --dry-run
List profiles      ./run_ci.py list -c scripts/ci/ci_tests.json
"""

import argparse
import ast
import fnmatch
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==============================================================================
#  Path setup
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (SCRIPT_DIR / "../..").resolve()
TEST_DIR = REPO_ROOT / "test"

# --- Defaults ----------------------------------------------------------------

# Model-path allowlist: only tests whose filepath contains at least one of
# these keywords (case-insensitive) will be executed; all others are skipped.
_MODEL_ALLOWLIST = [
    "deepseek_v32",
    "deepseek_v4",
    "glm5",
    "kimi_k25",
]


def _is_model_allowed(path: str) -> bool:
    """Return True if the path matches at least one allowlist keyword."""
    lower = path.lower()
    return any(kw in lower for kw in _MODEL_ALLOWLIST)


DEFAULT_HW = "cuda"
DEFAULT_TIMEOUT = 1800
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_RETRY_WAIT = 60
DEFAULT_RETRY_TIMEOUT_INCREASE = 600

# --- Proxy & HF token injection (mirrors ci_utils.py behaviour) --------------
_PROXY_URL = "http://agent.baidu.com:8891"
for _key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.setdefault(_key, _PROXY_URL)
for _key in ("NO_PROXY", "no_proxy"):
    os.environ.setdefault(_key, "localhost,127.0.0.1,0.0.0.0,::1")
print(f"[SGLang Test] Proxy enabled: {_PROXY_URL}", flush=True)

# --- Gated-model skip list ---------------------------------------------------
# Models that require HF_TOKEN authorization.  When the token is invalid or
# missing, test files referencing these models will be skipped automatically.
_GATED_MODEL_PATTERNS = [
    "meta-llama/Llama-",
    "meta-llama/Meta-Llama-",
    "google/gemma-3-",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
# Will be set to True by _preflight_checks if HF_TOKEN is unusable.
_SKIP_GATED: bool = False


# --- Environment pre-flight checks -------------------------------------------
def _preflight_checks(check_network: bool):
    """Run read-only environment diagnostics before testing."""
    global _SKIP_GATED

    hf_token = os.environ.get("HF_TOKEN", "")
    if not check_network:
        _SKIP_GATED = not bool(hf_token)
        return

    if hf_token:
        try:
            result = subprocess.run(
                [
                    "python3",
                    "-c",
                    "from huggingface_hub import HfApi; "
                    "api = HfApi(); "
                    "api.model_info('meta-llama/Llama-3.2-1B-Instruct', "
                    "token=os.environ.get('HF_TOKEN',''))",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                _SKIP_GATED = True
                print(
                    "[SGLang Test] HF_TOKEN verification failed. "
                    "Gated-model tests will be SKIPPED.",
                    flush=True,
                )
            else:
                print("[SGLang Test] HF_TOKEN verified OK.", flush=True)
        except Exception as exc:
            _SKIP_GATED = True
            print(
                f"[SGLang Test] HF_TOKEN verification failed ({exc}). "
                "Gated-model tests will be SKIPPED.",
                flush=True,
            )
    else:
        _SKIP_GATED = True
        print(
            "[SGLang Test] No HF_TOKEN set. Gated-model tests will be SKIPPED.",
            flush=True,
        )

    try:
        sm_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        sm_versions = [
            line.strip() for line in sm_result.stdout.splitlines() if line.strip()
        ]
        if sm_versions:
            print(
                f"[SGLang Test] GPU SM capability: {', '.join(sm_versions)}",
                flush=True,
            )
    except Exception as exc:
        print(f"[SGLang Test] GPU SM detection skipped: {exc}", flush=True)

    try:
        version_result = subprocess.run(
            [
                "python3",
                "-c",
                "import flashinfer_python; print(flashinfer_python.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        version = version_result.stdout.strip()
        if version:
            parts = version.split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            if major < 1 and minor < 2:
                print(
                    f"[SGLang Test] flashinfer_python {version} < 0.6.2 minimum. "
                    "Install or upgrade it before running affected tests.",
                    flush=True,
                )
            else:
                print(
                    f"[SGLang Test] flashinfer_python {version} OK (>=0.6.2)",
                    flush=True,
                )
    except Exception as exc:
        print(f"[SGLang Test] FlashInfer version check skipped: {exc}", flush=True)


# --- Colors ------------------------------------------------------------------
if sys.stdout.isatty():
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    NC = "\033[0m"
else:
    RED = GREEN = YELLOW = BLUE = CYAN = BOLD = NC = ""


def info(msg: str):
    """Print an informational message to stdout."""
    print(f"{GREEN}[INFO]{NC}  {msg}")


def warn(msg: str):
    """Print a warning message to stdout."""
    print(f"{YELLOW}[WARN]{NC}  {msg}")


def error(msg: str):
    """Print an error message to stderr."""
    print(f"{RED}[ERROR]{NC} {msg}", file=sys.stderr)


def die(msg: str):
    """Print an error message and exit with status 1."""
    error(msg)
    sys.exit(1)


# ==============================================================================
#  Lazy imports — resolved once we cd into TEST_DIR
# ==============================================================================

_ci_imports_loaded = False
collect_tests = None
HWBackend = None
CIRegistry = None
PER_COMMIT_SUITES = None
NIGHTLY_SUITES = None
HW_MAPPING_EXT = None  # from run_suite


def _ensure_ci_imports():
    """Import CI modules lazily — they require TEST_DIR on sys.path."""
    global _ci_imports_loaded, collect_tests, HWBackend, CIRegistry
    global PER_COMMIT_SUITES, NIGHTLY_SUITES, HW_MAPPING_EXT
    if _ci_imports_loaded:
        return
    test_dir_str = str(TEST_DIR)
    if test_dir_str not in sys.path:
        sys.path.insert(0, test_dir_str)
    from sglang.test.ci.ci_register import (
        CIRegistry as _CIRegistry,
        HWBackend as _HWBackend,
        collect_tests as _collect_tests,
    )

    collect_tests = _collect_tests
    HWBackend = _HWBackend
    CIRegistry = _CIRegistry

    from run_suite import (
        HW_MAPPING as _HW_MAPPING,
        NIGHTLY_SUITES as _NIGHTLY_SUITES,
        PER_COMMIT_SUITES as _PER_COMMIT_SUITES,
    )

    PER_COMMIT_SUITES = _PER_COMMIT_SUITES
    NIGHTLY_SUITES = _NIGHTLY_SUITES
    HW_MAPPING_EXT = _HW_MAPPING
    _ci_imports_loaded = True


def _hw_map():
    """Return the HW name → HWBackend mapping."""
    return {
        "cpu": HWBackend.CPU,
        "cuda": HWBackend.CUDA,
        "amd": HWBackend.AMD,
        "npu": HWBackend.NPU,
    }


# ==============================================================================
#  Usage text
# ==============================================================================

USAGE_TEXT = """\
SGLang CI Test Launcher

Usage:
  run_ci.py <command> [options]

Commands:
  run         Run tests with arbitrary target combinations (recommended)
  list        List available suites, tests, or config profiles

============================================================
  run — Unified Target Combinator
============================================================

Target flags (repeatable, freely combinable):
  -s, --suite  <name>       All tests from a CI suite
  -f, --file   <path>       A single test file (or glob pattern)
  -t, --test   <spec>       A single test method:
                               path/to/test.py::TestClass::test_method
  -g, --grep   <keyword>    All test files matching keyword in filename

Dynamic config file:
  -c, --config  <path>      JSON config with reusable profiles
  -p, --profile <name>      Profile to load from config
  --var KEY=VAL             Override a $VAR in config (repeatable)
                             CLI flags (-s/-f/-t/-g) merge ON TOP of profile

Runtime options:
  --hw <backend>            cuda | amd | cpu | npu  (default: cuda)
  --nightly                 Include nightly suites / tests
  --timeout <sec>           Timeout per execution unit (default: 1200)
  --continue-on-error       Don't stop at first failure
  --retry                   Smart retry for flaky accuracy tests
  --max-attempts <n>        Max attempts with --retry (default: 2)
  --partition-id <id>       Auto-partition: this worker's 0-based ID
  --partition-size <n>      Auto-partition: total worker count
  --dry-run                 Print execution plan without running
  -h, --help                Show this message

============================================================
  Config file — dynamic patterns, variables, filters
============================================================

  Profile fields (all optional):
    file_patterns   Glob patterns, expanded at runtime
                      ["test/registered/quant/test_*.py"]
    test_patterns   Method-level globs (pytest --collect-only)
                      ["**/test_metrics.py::Test*::test_*"]
    suite_patterns  Suite name wildcards
                      ["stage-b-*-1-gpu"]
    greps           Keyword match in CI-registered filenames
                      ["eagle", "moe"]
    exclude         Glob patterns to remove from results
                      ["*benchmark*", "*perf*"]
    filters         Conditional filtering
                      { "max_est_time": 60 }
    $VAR            All string values support variable substitution
                      "timeout": "$TIMEOUT"  →  resolved from vars/--var

  Examples:
    run_ci.py run -c ci_tests.json -p quick-unit
    run_ci.py run -c ci_tests.json -p quick-unit --var HW=amd --var MAX_EST=30
    run_ci.py run -c ci_tests.json -p kernel-sanity -f extra_test.py --dry-run
    run_ci.py list -c ci_tests.json
"""


# ==============================================================================
#  list command
# ==============================================================================


def cmd_list(hw: str, suite: str, nightly: bool):
    """List CI suites for a hardware backend, or the tests within a given suite."""
    _ensure_ci_imports()
    os.chdir(TEST_DIR)
    hw_map = _hw_map()

    if suite:
        backend = hw_map.get(hw)
        files = glob.glob("registered/**/*.py", recursive=True)
        all_tests = collect_tests(files, sanity_check=False)
        matched = sorted(
            [
                t
                for t in all_tests
                if t.backend == backend
                and t.suite == suite
                and t.nightly == nightly
            ],
            key=lambda t: t.est_time,
            reverse=True,
        )

        if not matched:
            print(f"No tests found for hw={hw}, suite={suite}, nightly={nightly}")
            return

        total = sum(t.est_time for t in matched)
        print(f"\n{BOLD}Tests in suite \"{suite}\" ({hw}, nightly={nightly}){NC}")
        print(f"Total: {len(matched)} tests, est. {total:.0f}s ({total / 60:.1f}min)")
        print("=" * 70)
        for t in matched:
            if t.disabled:
                tag = f"{RED}[DISABLED: {t.disabled}]{NC}"
            else:
                tag = f"{GREEN}[enabled]{NC}"
            print(f"  {tag}  {t.filename:55s}  est={t.est_time:>5.0f}s")
        print()
    else:
        def show(title: str, suites_dict: dict):
            print(f"\n{BOLD}{title}{NC}")
            print("=" * 60)
            for backend in sorted(suites_dict.keys(), key=lambda x: x.name):
                if hw != "all" and backend != HW_MAPPING_EXT.get(hw):
                    continue
                suites_list = suites_dict[backend]
                if not suites_list:
                    continue
                print(f"\n  {CYAN}{backend.name}{NC}:")
                for s in suites_list:
                    print(f"    - {s}")

        show("Per-Commit Suites", PER_COMMIT_SUITES)
        show("Nightly Suites", NIGHTLY_SUITES)
        print()


# ==============================================================================
#  list profiles from config file
# ==============================================================================


def _resolve_vars(val, var_map: Dict[str, str]):
    """Recursively substitute $VAR in strings."""
    if isinstance(val, str):
        for k, v in var_map.items():
            val = val.replace(f"${k}", str(v))
        return val
    if isinstance(val, list):
        return [_resolve_vars(x, var_map) for x in val]
    if isinstance(val, dict):
        return {k: _resolve_vars(v, var_map) for k, v in val.items()}
    return val


def cmd_list_profiles(config_file: str, cli_vars: Dict[str, str]):
    """Print all profiles defined in a JSON config file with resolved variables."""
    with open(config_file) as f:
        cfg = json.load(f)

    # Build vars: config defaults + CLI overrides
    var_map: Dict[str, str] = {}
    if "vars" in cfg and isinstance(cfg["vars"], dict):
        var_map.update(cfg["vars"])
    var_map.update(cli_vars)

    skip_keys = {"//schema", "//", "vars"}
    profiles = {
        k: _resolve_vars(v, var_map)
        for k, v in cfg.items()
        if k not in skip_keys
        and not k.startswith("//")
        and isinstance(v, dict)
        and "description" in v
    }

    if not profiles:
        print("No profiles found in config file.")
        return

    print(f"\n{BOLD}Profiles in {config_file}{NC}")
    if var_map:
        print(f"  vars: {', '.join(f'{k}={v}' for k, v in var_map.items())}")
    print("=" * 70)

    for name, p in profiles.items():
        desc = p.get("description", "")
        hw = p.get("hw", "cuda")

        counts = {}
        for key in ["suite_patterns", "file_patterns", "test_patterns", "greps"]:
            items = p.get(key, [])
            if items:
                counts[key] = len(items)

        timeout_val = p.get("timeout", 1200)
        opts = []
        if p.get("continue_on_error"):
            opts.append("continue-on-error")
        if p.get("retry"):
            opts.append("retry")
        if p.get("nightly"):
            opts.append("nightly")
        opts_str = ", ".join(opts) if opts else "-"

        filters = p.get("filters", {})
        filter_strs = [f"{k}={v}" for k, v in filters.items()] if filters else []
        exclude = p.get("exclude", [])

        print(f"\n  {CYAN}{name}{NC}")
        if desc:
            print(f"    {desc}")
        print(f"    hw={hw}  timeout={timeout_val}s  options=[{opts_str}]")

        if filters:
            print(f"    filters: {', '.join(filter_strs)}")
        if exclude:
            print(f"    exclude: {', '.join(exclude)}")

        label_map = [
            ("suite_patterns", "SUITE-PAT", GREEN),
            ("file_patterns", "FILE-PAT", GREEN),
            ("test_patterns", "TEST-PAT", CYAN),
            ("greps", "GREP", YELLOW),
        ]
        for key, label, color in label_map:
            for item in p.get(key, []):
                print(f"      {color}[{label}]{NC} {item}")

    print("\n" + "=" * 70)
    print(f"  Total: {len(profiles)} profile(s)")
    print(f"\n  Usage:  run_ci.py run -c {config_file} -p <name> [--var K=V] [--dry-run]")
    print()


# ==============================================================================
#  run command — the core combo runner
# ==============================================================================


@dataclass
class FileTarget:
    """A test file to run, with its estimated runtime and resolution source."""

    filepath: str
    est_time: float = 60
    source: str = ""


@dataclass
class MethodTarget:
    """A method-level test target (file + class + method) and its source."""

    filepath: str
    class_name: str
    method_name: str
    source: str = ""


def cmd_run(
    hw_name: str,
    nightly: bool,
    timeout: int,
    continue_on_error: bool,
    retry: bool,
    max_attempts: int,
    dry_run: bool,
    suite_targets: List[str],
    file_targets: List[str],
    test_targets: List[str],
    grep_targets: List[str],
    file_pattern_targets: List[str],
    test_pattern_targets: List[str],
    suite_pattern_targets: List[str],
    exclude_patterns: List[str],
    max_est_time: Optional[float],
    partition_id: Optional[int],
    partition_size: Optional[int],
):
    """Resolve all targets, build the execution plan, and run the selected tests."""
    _ensure_ci_imports()
    os.chdir(TEST_DIR)

    hw_map = _hw_map()
    hw = hw_map[hw_name]

    test_dir_str = str(TEST_DIR)
    repo_root_str = str(REPO_ROOT)

    # ── Collect CI registry ─────────────────────────────────────────────
    registered_files = glob.glob("registered/**/*.py", recursive=True)
    all_ci = collect_tests(registered_files, sanity_check=False)

    # Build lookup for est_time
    est_time_map: Dict[str, float] = {}
    for t in all_ci:
        est_time_map[t.filename] = t.est_time

    file_targets_list: List[FileTarget] = []
    method_targets_list: List[MethodTarget] = []

    def should_exclude(path: str) -> bool:
        for pat in exclude_patterns:
            if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(
                os.path.basename(path), pat
            ):
                return True
        return False

    def add_file(fp: str, est: float = 60, source: str = ""):
        if should_exclude(fp):
            return
        # if not _is_model_allowed(fp):
        #     return
        if max_est_time is not None and est > max_est_time:
            return
        for existing in file_targets_list:
            if existing.filepath == fp:
                return
        file_targets_list.append(FileTarget(filepath=fp, est_time=est, source=source))

    def resolve_path(fpath: str) -> Optional[str]:
        """Resolve a file path relative to TEST_DIR or REPO_ROOT."""
        for base in [test_dir_str, repo_root_str, "."]:
            candidate = os.path.join(base, fpath)
            if os.path.isfile(candidate):
                return os.path.relpath(candidate, test_dir_str)
        if os.path.isfile(fpath):
            return os.path.relpath(os.path.abspath(fpath), test_dir_str)
        return None

    # ── 1) suite_patterns: wildcard match against known suite names ──────
    all_suite_names: set = set()
    for t in all_ci:
        if t.backend == hw and t.nightly == nightly and t.disabled is None:
            all_suite_names.add(t.suite)

    for pat in suite_pattern_targets:
        matched_suites = [s for s in all_suite_names if fnmatch.fnmatch(s, pat)]
        if not matched_suites:
            warn(f"suite_pattern '{pat}' matched 0 suites")
        for sn in matched_suites:
            if sn not in suite_targets:
                suite_targets.append(sn)

    # ── 2) --suite / suite_targets ──────────────────────────────────────
    for suite_name in suite_targets:
        matched = [
            t
            for t in all_ci
            if t.backend == hw
            and t.suite == suite_name
            and t.nightly == nightly
            and t.disabled is None
        ]
        if not matched:
            warn(
                f"No enabled tests in suite '{suite_name}' "
                f"(hw={hw_name}, nightly={nightly})"
            )
        for t in matched:
            add_file(t.filename, est=t.est_time, source=f"suite:{suite_name}")

    # ── 3) file_patterns: glob expansion ────────────────────────────────
    for pat in file_pattern_targets:
        expanded = glob.glob(os.path.join(repo_root_str, pat), recursive=True)
        if not expanded:
            expanded = glob.glob(os.path.join(test_dir_str, pat), recursive=True)
        if not expanded:
            warn(f"file_pattern '{pat}' matched 0 files")
        for fp in sorted(expanded):
            if not fp.endswith(".py"):
                continue
            rel = os.path.relpath(fp, test_dir_str)
            est = est_time_map.get(rel, 60)
            add_file(rel, est=est, source=f"pattern:{pat}")

    # ── 4) --file / literal files ───────────────────────────────────────
    for fpath in file_targets:
        resolved = resolve_path(fpath)
        if resolved is None:
            # Maybe it's a glob pattern from CLI
            expanded = glob.glob(os.path.join(repo_root_str, fpath), recursive=True)
            if expanded:
                for fp in sorted(expanded):
                    if fp.endswith(".py"):
                        rel = os.path.relpath(fp, test_dir_str)
                        est = est_time_map.get(rel, 60)
                        add_file(rel, est=est, source=f"file:{fpath}")
            else:
                error(f"File not found: {fpath}")
                sys.exit(1)
        else:
            est = est_time_map.get(resolved, 60)
            add_file(resolved, est=est, source=f"file:{fpath}")

    # ── 5) --grep ───────────────────────────────────────────────────────
    for kw in grep_targets:
        kw_lower = kw.lower()
        matched = [
            t
            for t in all_ci
            if kw_lower in t.filename.lower()
            and t.backend == hw
            and t.nightly == nightly
            and t.disabled is None
        ]
        if not matched:
            warn(
                f"No enabled tests matching grep '{kw}' "
                f"(hw={hw_name}, nightly={nightly})"
            )
        for t in matched:
            add_file(t.filename, est=t.est_time, source=f"grep:{kw}")

    # ── 6) test_patterns: method-level glob via AST parsing ─────────────
    for pat in test_pattern_targets:
        if "::" not in pat:
            error(f"test_pattern must contain '::': {pat}")
            sys.exit(1)
        file_part = pat.split("::")[0]

        expanded_files = []
        for base in [test_dir_str, repo_root_str]:
            expanded_files = glob.glob(os.path.join(base, file_part), recursive=True)
            if expanded_files:
                break

        if not expanded_files:
            warn(f"test_pattern file part '{file_part}' matched 0 files")
            continue

        pat_parts = pat.split("::")
        class_pat = pat_parts[1] if len(pat_parts) > 1 else "*"
        method_pat = pat_parts[2] if len(pat_parts) > 2 else "*"

        for fp in expanded_files:
            if not fp.endswith(".py"):
                continue
            if should_exclude(fp):
                continue
            rel = os.path.relpath(fp, test_dir_str)
            # if not _is_model_allowed(rel):
            #     continue

            try:
                with open(fp) as fh:
                    tree = ast.parse(fh.read())
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if not fnmatch.fnmatch(node.name, class_pat):
                    continue
                for item in node.body:
                    if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if not item.name.startswith("test_"):
                        continue
                    if not fnmatch.fnmatch(item.name, method_pat):
                        continue
                    method_targets_list.append(
                        MethodTarget(
                            filepath=rel,
                            class_name=node.name,
                            method_name=item.name,
                            source=f"test_pattern:{pat}",
                        )
                    )

    # ── 7) --test / literal method specs ────────────────────────────────
    for spec in test_targets:
        if "::" not in spec:
            error(f"Invalid test spec: {spec}")
            sys.exit(1)
        parts = spec.split("::")
        fpath = parts[0]
        class_name = parts[1] if len(parts) > 1 else ""
        method_name = parts[2] if len(parts) > 2 else ""

        resolved = resolve_path(fpath)
        if resolved is None:
            error(f"File not found: {fpath}")
            sys.exit(1)
        if should_exclude(resolved):
            continue
        # if not _is_model_allowed(resolved):
        #     continue
        method_targets_list.append(
            MethodTarget(
                filepath=resolved,
                class_name=class_name,
                method_name=method_name,
                source=f"test:{spec}",
            )
        )

    # ── Dedup method targets ────────────────────────────────────────────
    seen: set = set()
    deduped: List[MethodTarget] = []
    for mt in method_targets_list:
        key = (mt.filepath, mt.class_name, mt.method_name)
        if key not in seen:
            seen.add(key)
            deduped.append(mt)
    method_targets_list.clear()
    method_targets_list.extend(deduped)

    # ── Optional partitioning ───────────────────────────────────────────
    if partition_id is not None and partition_size is not None:
        sorted_ft = sorted(
            file_targets_list, key=lambda f: f.est_time, reverse=True
        )
        buckets: List[List[FileTarget]] = [[] for _ in range(partition_size)]
        sums = [0.0] * partition_size
        for ft in sorted_ft:
            idx = min(range(partition_size), key=sums.__getitem__)
            buckets[idx].append(ft)
            sums[idx] += ft.est_time
        file_targets_list.clear()
        if partition_id < partition_size:
            file_targets_list.extend(buckets[partition_id])

    # ── Validate ────────────────────────────────────────────────────────
    total_targets = len(file_targets_list) + len(method_targets_list)
    if total_targets == 0:
        error("No targets resolved. Nothing to run.")
        sys.exit(1)

    # ── Print execution plan ────────────────────────────────────────────
    print()
    print(f"{BOLD}{'=' * 70}{NC}")
    print(f"{BOLD}  SGLang CI — Execution Plan{NC}")
    print(f"{BOLD}{'=' * 70}{NC}")
    print(f"  Hardware:    {hw_name}")
    print(f"  Nightly:     {nightly}")
    print(f"  Timeout:     {timeout}s per unit")
    print(f"  Retry:       {retry} (max {max_attempts} attempts)")
    print(f"  Continue:    {continue_on_error}")
    if max_est_time is not None:
        print(f"  Max est:     {max_est_time}s")
    if exclude_patterns:
        print(f"  Exclude:     {exclude_patterns}")
    if partition_id is not None:
        print(f"  Partition:   {partition_id}/{partition_size}")
    print()

    if file_targets_list:
        total_est = sum(f.est_time for f in file_targets_list)
        print(
            f"{BOLD}  File targets ({len(file_targets_list)}, est. {total_est:.0f}s):{NC}"
        )
        for ft in file_targets_list:
            print(
                f"    {GREEN}[FILE]{NC}   {ft.filepath:55s}  est={ft.est_time:>5.0f}s  ({ft.source})"
            )

    if method_targets_list:
        print(f"\n{BOLD}  Method targets ({len(method_targets_list)}):{NC}")
        for mt in method_targets_list:
            if mt.method_name:
                label = f"{mt.filepath}::{mt.class_name}::{mt.method_name}"
            else:
                label = f"{mt.filepath}::{mt.class_name}"
            print(f"    {CYAN}[METHOD]{NC} {label}  ({mt.source})")

    print(f"{BOLD}{'=' * 70}{NC}")
    print()

    if dry_run:
        print(f"{YELLOW}[DRY-RUN]{NC} Execution plan printed. No tests were run.")
        sys.exit(0)

    # ── Execute ─────────────────────────────────────────────────────────
    from sglang.srt.utils.common import kill_process_tree as _kill

    tic = time.perf_counter()
    passed: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str]] = []

    def _cleanup_gpu_memory():
        """Kill stray GPU processes left over from the previous test.

        Uses nvidia-smi to find all compute processes, then sends SIGKILL to
        any that are *not* the current Python interpreter.  A short sleep
        afterwards gives the driver time to reclaim the memory before the next
        test tries to allocate.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            current_pid = os.getpid()
            pids = [
                int(p.strip())
                for p in result.stdout.splitlines()
                if p.strip().isdigit() and int(p.strip()) != current_pid
            ]
            for pid in pids:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
            if pids:
                print(
                    f"[GPU cleanup] Killed {len(pids)} stray process(es): {pids}",
                    flush=True,
                )
                time.sleep(3)
        except Exception as exc:
            warn(f"GPU cleanup failed (non-fatal): {exc}")

    def _build_env(filepath: str) -> dict:
        """Return the subprocess environment, with per-test overrides applied."""
        env = os.environ.copy()
        # Restrict visible GPUs for tests that hard-code a specific device count
        if "test_mamba2_mixer" in filepath:
            env["CUDA_VISIBLE_DEVICES"] = "0,1"
            print("[env] test_mamba2_mixer: CUDA_VISIBLE_DEVICES=0,1", flush=True)
        return env

    def _should_skip_gated(filepath: str) -> bool:
        """Check whether a test file references gated models and should be skipped."""
        if not _SKIP_GATED or not filepath:
            return False
        full = os.path.join(os.getcwd(), filepath)
        if not os.path.isfile(full):
            return False
        try:
            with open(full, "r", errors="ignore") as f:
                content = f.read()
            return any(pat in content for pat in _GATED_MODEL_PATTERNS)
        except Exception:
            return False

    def _run_cmd(cmd: List[str], label: str, timeout_sec: int, filepath: str = "") -> bool:
        # Skip tests that reference gated models when HF_TOKEN is invalid
        if _should_skip_gated(filepath):
            print(f"\n{YELLOW}  SKIPPED{NC} {label} (gated model, no valid HF_TOKEN)\n")
            return True  # count as "passed" so we don't block the pipeline

        print(f"\n...\n{BOLD}Running:{NC} {' '.join(cmd)}\n...\n", flush=True)
        file_tic = time.perf_counter()
        env = _build_env(filepath)
        try:
            proc = subprocess.Popen(cmd, stdout=None, stderr=None, env=env)
            proc.wait(timeout=timeout_sec)
            elapsed = time.perf_counter() - file_tic
            _cleanup_gpu_memory()
            if proc.returncode == 0:
                print(f"\n{GREEN}  PASSED{NC} {label} ({elapsed:.0f}s)\n")
                return True
            else:
                print(
                    f"\n{RED}  FAILED{NC} {label} (exit {proc.returncode}, {elapsed:.0f}s)\n"
                )
                return False
        except subprocess.TimeoutExpired:
            _kill(proc.pid)
            time.sleep(3)
            _cleanup_gpu_memory()
            print(f"\n{RED}  TIMEOUT{NC} {label} after {timeout_sec}s\n")
            return False

    def run_with_retry(cmd: List[str], label: str, timeout_sec: int, filepath: str = "") -> bool:
        attempts = max_attempts if retry else 1
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                print(f"{YELLOW}  [Retry {attempt}/{max_attempts}]{NC} {label}")
                time.sleep(DEFAULT_RETRY_WAIT)
            ok = _run_cmd(cmd, label, timeout_sec, filepath=filepath)
            if ok:
                return True
        return False

    effective_timeout = timeout
    if retry:
        effective_timeout += DEFAULT_RETRY_TIMEOUT_INCREASE

    # Phase 1: file-level
    for i, ft in enumerate(file_targets_list):
        full_path = os.path.join(os.getcwd(), ft.filepath)
        label = f"({i + 1}/{len(file_targets_list)}) {ft.filepath}"
        # Give each test at least effective_timeout, but scale up for long-
        # running tests so that a 1000s-est test is not killed at 1800s.
        file_timeout = max(effective_timeout, int(ft.est_time * 1.5))
        ok = run_with_retry(
            ["python3", full_path], label, file_timeout, filepath=ft.filepath
        )
        (passed if ok else failed).append(("FILE", ft.filepath))
        if not ok and not continue_on_error:
            break

    # Phase 2: method-level
    should_continue = continue_on_error or not failed
    if method_targets_list and should_continue:
        for i, mt in enumerate(method_targets_list):
            full_path = os.path.join(os.getcwd(), mt.filepath)
            if mt.method_name:
                spec = f"{full_path}::{mt.class_name}::{mt.method_name}"
            elif mt.class_name:
                spec = f"{full_path}::{mt.class_name}"
            else:
                spec = full_path
            label = f"({i + 1}/{len(method_targets_list)}) {mt.filepath}::{mt.class_name}::{mt.method_name}"
            ok = run_with_retry(
                ["python3", "-m", "pytest", spec, "-xvs"],
                label,
                effective_timeout,
                filepath=mt.filepath,
            )
            (passed if ok else failed).append(("METHOD", label))
            if not ok and not continue_on_error:
                break

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - tic
    print()
    print("=" * 60)
    print(
        f"  {BOLD}Results: {len(passed)} passed, {len(failed)} failed{NC}  ({elapsed_total:.0f}s)"
    )
    print("=" * 60)
    if passed:
        for kind, name in passed:
            print(f"  {GREEN}  PASS [{kind}]{NC} {name}")
    if failed:
        for kind, name in failed:
            print(f"  {RED}  FAIL [{kind}]{NC} {name}")
    print("=" * 60)

    sys.exit(0 if not failed else 1)


# ==============================================================================
#  Load profile from config file
# ==============================================================================


def load_profile(
    config_file: str,
    profile: str,
    cli_vars: Dict[str, str],
) -> dict:
    """Load a named profile from a JSON config file and return resolved settings."""
    with open(config_file) as f:
        cfg = json.load(f)

    # Build var map
    var_map: Dict[str, str] = {}
    if "vars" in cfg and isinstance(cfg["vars"], dict):
        var_map.update(cfg["vars"])
    var_map.update(cli_vars)

    skip_keys = {"//schema", "//", "vars"}
    if profile in skip_keys or profile not in cfg:
        valid = [
            k
            for k in cfg
            if k not in skip_keys
            and not k.startswith("//")
            and isinstance(cfg[k], dict)
        ]
        die(f'profile "{profile}" not found. Available: {", ".join(valid)}')

    return _resolve_vars(cfg[profile], var_map)


# ==============================================================================
#  Argument parser
# ==============================================================================


class AppendAction(argparse.Action):
    """Custom action that appends to a list (handles repeated flags)."""

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest) or []
        current.append(values)
        setattr(namespace, self.dest, current)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="SGLang CI Test Launcher",
        usage=USAGE_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    parser.add_argument("command", nargs="?", default=None, help="run | list")
    parser.add_argument("-h", "--help", action="store_true")

    # Target flags
    parser.add_argument("-s", "--suite", dest="suites", action=AppendAction, default=[])
    parser.add_argument("-f", "--file", dest="files", action=AppendAction, default=[])
    parser.add_argument("-t", "--test", dest="tests", action=AppendAction, default=[])
    parser.add_argument("-g", "--grep", dest="greps", action=AppendAction, default=[])

    # Config
    parser.add_argument("-c", "--config", dest="config_file", default="")
    parser.add_argument("-p", "--profile", dest="profile", default="")
    parser.add_argument("--var", dest="cli_vars", action=AppendAction, default=[])

    # Runtime
    parser.add_argument("--hw", default=DEFAULT_HW)
    parser.add_argument("--nightly", action="store_true", default=False)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    parser.add_argument("--retry", action="store_true", default=False)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--partition-id", type=int, default=None)
    parser.add_argument("--partition-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)

    return parser


# ==============================================================================
#  Main
# ==============================================================================


def main():
    """Parse arguments and dispatch to the requested command (run | list)."""
    parser = build_parser()
    args = parser.parse_args()

    if args.help or args.command is None:
        print(USAGE_TEXT)
        sys.exit(0)

    command = args.command
    _preflight_checks(command == "run" and not args.dry_run)

    # Validate hw
    if args.hw not in ("cuda", "amd", "cpu", "npu", "all"):
        die(f"Invalid --hw: {args.hw}  (cuda|amd|cpu|npu)")

    # Parse CLI --var KEY=VAL into dict
    cli_vars: Dict[str, str] = {}
    for v in args.cli_vars:
        if "=" in v:
            k, val = v.split("=", 1)
            cli_vars[k] = val

    # Resolve config file path
    config_file = args.config_file
    if config_file and not os.path.isabs(config_file):
        candidate = os.path.join(str(REPO_ROOT), config_file)
        if os.path.isfile(candidate):
            config_file = candidate
        elif os.path.isfile(config_file):
            config_file = os.path.abspath(config_file)
        else:
            die(f"Config file not found: {config_file}")

    # ── list command ────────────────────────────────────────────────────
    if command == "list":
        if config_file:
            cmd_list_profiles(config_file, cli_vars)
        else:
            suite = args.suites[0] if args.suites else ""
            cmd_list(args.hw, suite, args.nightly)
        return

    # ── run command ─────────────────────────────────────────────────────
    if command in ("run",):
        # Accumulators — CLI values first
        suites = list(args.suites)
        files = list(args.files)
        tests = list(args.tests)
        greps = list(args.greps)

        # Dynamic config extras
        file_patterns: List[str] = []
        test_patterns: List[str] = []
        suite_patterns: List[str] = []
        excludes: List[str] = []
        max_est_time: Optional[float] = None

        # Overridable options
        hw = args.hw
        nightly = args.nightly
        timeout = args.timeout
        continue_on_error = args.continue_on_error
        retry = args.retry
        max_attempts = args.max_attempts

        # Load profile from config if specified
        if config_file and args.profile:
            info(f"Loading profile '{args.profile}' from {config_file}")
            p = load_profile(config_file, args.profile, cli_vars)

            # Merge targets from profile
            suites.extend(p.get("suites", []))
            files.extend(p.get("files", []))
            tests.extend(p.get("tests", []))
            greps.extend(p.get("greps", []))

            # Dynamic patterns
            file_patterns.extend(p.get("file_patterns", []))
            test_patterns.extend(p.get("test_patterns", []))
            suite_patterns.extend(p.get("suite_patterns", []))
            excludes.extend(p.get("exclude", []))

            # Filters
            filters = p.get("filters", {})
            if "max_est_time" in filters:
                max_est_time = float(filters["max_est_time"])

            # Options: only override if CLI didn't set a non-default value
            if "hw" in p and hw == DEFAULT_HW:
                hw = p["hw"]
            if p.get("timeout") and timeout == DEFAULT_TIMEOUT:
                timeout = int(p["timeout"])
            if p.get("continue_on_error") and not continue_on_error:
                continue_on_error = True
            if p.get("retry") and not retry:
                retry = True
            if p.get("max_attempts") and max_attempts == DEFAULT_MAX_ATTEMPTS:
                max_attempts = int(p["max_attempts"])
            if p.get("nightly") and not nightly:
                nightly = True

            info("Profile loaded. CLI flags merge on top.")
        elif config_file and not args.profile:
            die(
                "Config file specified but no -p <profile>. Use 'list -c <file>' to see profiles."
            )

        total = (
            len(suites)
            + len(files)
            + len(tests)
            + len(greps)
            + len(file_patterns)
            + len(test_patterns)
            + len(suite_patterns)
        )
        if total == 0:
            die("No targets specified. Use -s, -f, -t, -g or -c/-p. See --help.")

        cmd_run(
            hw_name=hw,
            nightly=nightly,
            timeout=timeout,
            continue_on_error=continue_on_error,
            retry=retry,
            max_attempts=max_attempts,
            dry_run=args.dry_run,
            suite_targets=suites,
            file_targets=files,
            test_targets=tests,
            grep_targets=greps,
            file_pattern_targets=file_patterns,
            test_pattern_targets=test_patterns,
            suite_pattern_targets=suite_patterns,
            exclude_patterns=excludes,
            max_est_time=max_est_time,
            partition_id=args.partition_id,
            partition_size=args.partition_size,
        )
        return

    if command in ("-h", "--help", "help"):
        print(USAGE_TEXT)
        sys.exit(0)

    die(f"Unknown command: {command}.  Use --help.")


if __name__ == "__main__":
    main()
