#!/usr/bin/env python3
"""
RepoLauncher 2.0 — Autopilot Bootstrapper for Apple Silicon (M3 Max 128 GB)
=======================================================================

Purpose
-------
Clone *any* public (or private, with SSH) GitHub repository and bring it to a
*running* state on macOS 14 (Sequoia) with minimal user input and cost‑efficient
LLM guidance.

This script embodies the architecture defined in the accompanying PRD and
project docs. Everything is **heavily commented** to make maintenance and future
extension trivial.

High‑level Pipeline
-------------------
1. **Credential Orchestration** — Ensure `OPENAI_API_KEY` is available via
   Keychain or environment; prompt once if missing.
2. **Repo Acquisition**        — Clone or update target Git repo.
3. **Repo Inspection**         — Detect project type & ML frameworks.
4. **Environment Solver**      — Create isolated *venv* and install the optimal
   dependency set (Metal/MLX wheels, etc.).
5. **Hardware Profiling**      — Query ANE/TGPU cores and 128 GB unified memory
   to tune subsequent steps.
6. **Install / Run**           — Execute build or launch commands; capture
   stderr.
7. **LLM Advisor**             — On failure, send condensed trace to OpenAI via
   the *Responses API* to get deterministic patch suggestions (o4‑mini → 4o).
8. **Autonomous Retry**        — Apply suggestions; loop ≤ 3 times or until
   success / budget cap hit.
9. **Launch**                  — Auto‑open app (e.g., localhost URL) on success.

Note: Many subsystems are *scaffolds* (TODO marks) awaiting full
implementation—this file establishes the skeleton so that each task from
`tasks.md` maps to a clearly delimited section below.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Optional

# Optional / soft dependencies – imported lazily when first used
try:
    import keyring  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    keyring = None  # pyright: ignore[reportGeneralTypeIssues]

try:
    import openai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    openai = None  # pyright: ignore[reportGeneralTypeIssues]

# ---------------------------------------------------------------------------
# Configurable constants (tweak in code or expose via CLI flags later)
# ---------------------------------------------------------------------------
LLM_DEFAULT_MODEL: Final[str] = "o4-mini"  # cost‑efficient default
LLM_FALLBACK_MODEL: Final[str] = "gpt-4o"  # escalated model for complex traces
MAX_RETRIES: Final[int] = 3                # autonomous attempts after failure
DEFAULT_COST_BUDGET_USD: Final[float] = 1.00  # per‑run OpenAI cost ceiling

# Estimated $/1K tokens (keep in sync with real billing)
MODEL_COSTS = {
    "o4-mini": 0.0005,  # USD per 1K tokens (example)
    "gpt-4o": 0.005,
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def run(cmd: List[str] | str, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Wrapper for :pyfunc:`subprocess.run` with pretty logging."""
    if isinstance(cmd, str):
        cmd_display = cmd
    else:
        cmd_display = " ".join(cmd)
    logging.debug("$ %s", cmd_display)
    return subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        cwd=cwd,
        text=True,
        check=False,
        capture_output=capture,
    )


# ---------------------------------------------------------------------------
# Credential Manager
# ---------------------------------------------------------------------------
class CredentialManager:
    """Fetch and persist secret credentials (OpenAI key for now)."""

    SERVICE_NAME: Final[str] = "RepoLauncher"

    @classmethod
    def get_openai_key(cls) -> str:
        """Return the OpenAI API key or raise *RuntimeError* if user declined."""
        # 1️⃣ Env var wins (allows overriding per‑shell)
        key = os.getenv("OPENAI_API_KEY")
        if key:
            logging.debug("Using OPENAI_API_KEY from environment.")
            return key

        # 2️⃣ Keychain fallback
        if keyring:
            key = keyring.get_password(cls.SERVICE_NAME, "OPENAI_API_KEY")
            if key:
                logging.debug("Loaded OPENAI_API_KEY from macOS Keychain.")
                return key

        # 3️⃣ Interactive prompt (single shot)
        try:
            key = input("Enter your OpenAI API key (skipped = abort): ").strip()
        except EOFError:
            key = ""
        if not key:
            raise RuntimeError("OpenAI API key is required but not provided.")

        if keyring:
            try:
                keyring.set_password(cls.SERVICE_NAME, "OPENAI_API_KEY", key)
                logging.debug("Saved OPENAI_API_KEY to Keychain.")
            except keyring.errors.KeyringError:  # type: ignore[attr-defined]
                logging.warning("Failed to save key to Keychain; continuing anyway.")
        return key


# ---------------------------------------------------------------------------
# Repo Acquisition & Inspection
# ---------------------------------------------------------------------------
@dataclass
class RepoInfo:
    path: Path
    url: str
    project_type: str  # e.g., "python-ml", "node", …
    ml_frameworks: List[str]


class RepoInspector:
    """Clone/update the repo and detect tech stack."""

    @staticmethod
    def clone_or_update(url: str, dest_root: Path = Path.cwd() / "repos") -> Path:
        dest_root.mkdir(parents=True, exist_ok=True)
        repo_name = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        dest = dest_root / repo_name
        if dest.exists():
            logging.info("Updating existing repo %s", dest)
            run(["git", "-C", str(dest), "pull"], capture=False)
        else:
            logging.info("Cloning %s into %s", url, dest)
            run(["git", "clone", url, str(dest)])
        return dest

    @staticmethod
    def detect_project_type(repo_path: Path) -> str:
        """Rudimentary heuristics to label project type."""
        if (repo_path / "pyproject.toml").exists() or (repo_path / "setup.py").exists():
            return "python-ml" if RepoInspector._has_ml_files(repo_path) else "python"
        if (repo_path / "package.json").exists():
            return "node"
        if any(repo_path.glob("*.rs")):
            return "rust"
        return "unknown"

    @staticmethod
    def _has_ml_files(repo_path: Path) -> bool:  # noqa: D401
        """Detect if repo imports PyTorch/JAX/etc."""
        pattern = re.compile(r"\b(torch|jax|tensorflow|transformers)\b")
        for py in repo_path.rglob("*.py"):
            try:
                txt = py.read_text(errors="ignore")
            except Exception:
                continue
            if pattern.search(txt):
                return True
        return False

    @staticmethod
    def detect_ml_frameworks(repo_path: Path) -> List[str]:
        frameworks: set[str] = set()
        mapping = {
            "torch": "pytorch",
            "jax": "jax",
            "tensorflow": "tensorflow",
            "transformers": "transformers",
        }
        for py in repo_path.rglob("*.py"):
            try:
                txt = py.read_text(errors="ignore")
            except Exception:
                continue
            for k, v in mapping.items():
                if re.search(fr"\b{k}\b", txt):
                    frameworks.add(v)
        return sorted(frameworks)

    @classmethod
    def inspect(cls, url: str) -> RepoInfo:
        path = cls.clone_or_update(url)
        ptype = cls.detect_project_type(path)
        frameworks = cls.detect_ml_frameworks(path)
        return RepoInfo(path=path, url=url, project_type=ptype, ml_frameworks=frameworks)


# ---------------------------------------------------------------------------
# Environment Solver (Python‑focused MVP)
# ---------------------------------------------------------------------------
class EnvSolver:
    """Create virtualenv, install packages with Apple‑Silicon optimisations."""

    def __init__(self, repo: RepoInfo):
        self.repo = repo
        self.venv_dir = repo.path / ".venv"

    def create_venv(self) -> None:
        if self.venv_dir.exists():
            logging.info("Removing pre‑existing venv to ensure clean state…")
            shutil.rmtree(self.venv_dir)
        logging.info("Creating virtualenv under %s", self.venv_dir)
        run([sys.executable, "-m", "venv", str(self.venv_dir)])

    # ---- package installation helpers ----
    def _pip(self, args: List[str]) -> None:
        run([str(self.venv_dir / "bin" / "pip"), *args])

    # ---- ML‑aware install logic ----
    def install_dependencies(self) -> None:
        """Install project requirements with Apple‑tuned wheels where possible."""
        # 1. Base requirements file
        req_file = None
        if (self.repo.path / "requirements.txt").exists():
            req_file = self.repo.path / "requirements.txt"
        elif (self.repo.path / "requirements" / "default.txt").exists():
            req_file = self.repo.path / "requirements" / "default.txt"

        if req_file:
            logging.info("Installing pinned requirements…) ")
            self._pip(["install", "-r", str(req_file)])

        # 2. ML frameworks post‑processing
        if "pytorch" in self.repo.ml_frameworks:
            logging.info("Installing nightly Metal‑accelerated PyTorch…")
            self._pip(
                [
                    "install",
                    "--pre",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/nightly/cpu",
                ]
            )
        if "jax" in self.repo.ml_frameworks:
            logging.info("Adding MLX & jax‑mlx shim…")
            self._pip(["install", "mlx", "jax-mlx==0.0.4"])
        # TODO: TensorFlow‑macOS, etc.


# ---------------------------------------------------------------------------
# Hardware Profiler (simple MVP)
# ---------------------------------------------------------------------------
@dataclass
class HWProfile:
    ram_gb: int
    ane_cores: int
    gpu_cores: int


class HardwareProfiler:
    @staticmethod
    def profile() -> HWProfile:
        """Return coarse hardware stats (macOS)."""
        # RAM via `sysctl` (bytes)
        ram_bytes = int(
            run(["sysctl", "-n", "hw.memsize"], capture=True).stdout.strip()
        )
        ram_gb = ram_bytes // (1024 ** 3)

        # GPU + ANE via `system_profiler` (slow but OK for one‑shot)
        sp = run(["system_profiler", "SPHardwareDataType", "SPiBridgeDataType"], capture=True).stdout
        ane_match = re.search(r"Apple Neural Engine: (\d+) cores", sp)
        gpu_match = re.search(r"GPU.*?(\d+) cores", sp, re.DOTALL)
        ane = int(ane_match.group(1)) if ane_match else 0
        gpu = int(gpu_match.group(1)) if gpu_match else 0
        return HWProfile(ram_gb=ram_gb, ane_cores=ane, gpu_cores=gpu)


# ---------------------------------------------------------------------------
# LLM Advisor (Responses API wrapper — skeleton)
# ---------------------------------------------------------------------------
class LLMAdvisor:
    def __init__(self, api_key: str, budget_usd: float = DEFAULT_COST_BUDGET_USD):
        if openai is None:
            raise RuntimeError("openai‑python is not installed. Run `pip install openai>=1.25`.")
        openai.api_key = api_key
        self.budget_remaining = budget_usd

    def _cost_for_tokens(self, tokens: int, model: str) -> float:
        return tokens / 1000 * MODEL_COSTS.get(model, 0.005)

    def get_suggestion(self, trace: str) -> str | None:
        """Return an LLM patch suggestion text or *None* if budget exhausted."""
        if self.budget_remaining <= 0:
            logging.warning("Budget exhausted; skipping LLM suggestion.")
            return None

        model = LLM_DEFAULT_MODEL
        tokens_est = max(100, len(trace) // 4)  # rough heuristic
        est_cost = self._cost_for_tokens(tokens_est + 500, model)  # 500 token reply
        if est_cost > self.budget_remaining and model != LLM_FALLBACK_MODEL:
            # escalate to cheaper model — here cheaper default already chosen
            pass

        logging.info("Querying %s (est. $%.4f, remaining $%.2f)…", model, est_cost, self.budget_remaining)

        response = openai.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "text"},
            messages=[
                {"role": "system", "content": "You are RepoLauncher, an elite build engineer."},
                {"role": "user", "content": textwrap.dedent(
                    f"""
                    The following error trace occurred during project setup on macOS 14 (M3 Max, 128 GB RAM). 
                    Provide **concise shell commands or patch diff** to fix the issue. 
                    Error trace:
                    ```
                    {trace.strip()[:4000]}
                    ```
                    """
                )},
            ],
        )
        suggestion = response.choices[0].message.content.strip()
        # Deduct cost (rough approximation)
        self.budget_remaining -= est_cost
        return suggestion


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def bootstrap(url: str, budget: float) -> None:
    """High‑level orchestration of the full pipeline."""
    api_key = CredentialManager.get_openai_key()
    advisor = LLMAdvisor(api_key, budget)

    repo = RepoInspector.inspect(url)
    logging.info("Detected project type=%s, ML frameworks=%s", repo.project_type, repo.ml_frameworks)

    env = EnvSolver(repo)
    env.create_venv()

    hw = HardwareProfiler.profile()
    logging.info("Hardware profile: %s", hw)

    # Install dependencies with retries
    retries = 0
    while True:
        try:
            env.install_dependencies()
            break  # success
        except Exception as e:  # noqa: BLE001
            retries += 1
            logging.error("Dependency installation failed: %s", e)
            if retries > MAX_RETRIES:
                raise SystemExit("Max retries exceeded.")
            suggestion = advisor.get_suggestion(trace=str(e))
            if suggestion:
                logging.info("LLM suggestion:\n%s", suggestion)
                # ⚠️  For security, we *print* suggestions rather than auto‑execute.
                input("Apply suggested commands manually and press Enter to retry…")
            else:
                raise SystemExit("No more budget; aborting.")

    logging.info("🏁 Setup complete. Activate venv and start hacking!")
    print(f"export PATH=\"{env.venv_dir}/bin:$PATH\"")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RepoLauncher 2.0 — zero‑config repo bootstrapper.")
    p.add_argument("url", help="GitHub repository URL")
    p.add_argument("--budget", type=float, default=DEFAULT_COST_BUDGET_USD, help="OpenAI cost ceiling in USD (default: 1.00)")
    p.add_argument("--verbose", "-v", action="count", default=0, help="Increase logging verbosity (‑v/‑vv)")
    return p.parse_args(argv)


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING - (10 * min(verbosity, 2))  # WARNING / INFO / DEBUG
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


if __name__ == "__main__":
    ns = parse_args()
    setup_logging(ns.verbose)
    try:
        bootstrap(ns.url, ns.budget)
    except Exception as exc:  # noqa: BLE001
        logging.critical("Fatal: %s", exc)
        sys.exit(1)
