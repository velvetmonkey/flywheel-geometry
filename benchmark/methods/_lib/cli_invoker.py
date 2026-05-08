"""Thin CLI invoker for benchmark LLM calls — uses subscription tooling
(`claude`, `codex`, `gemini`) instead of paid API endpoints.

Mirrors the invocation pattern the Roundtable MCP server uses to dispatch
to local CLI agents. Each call is a one-shot non-interactive prompt; the
function returns the response text plus a trace dict for reproducibility.

CLI calls have non-trivial startup overhead (~10-25s per call against the
provider backend). For batched workloads, use `cli_invoke_many` with a
modest concurrency level — typically 5-10 — to amortise that cost without
tripping subscription rate limits.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, TypeVar


@dataclass
class CLIResponse:
    text: str
    cli: str
    model: Optional[str]
    returncode: int
    wall_clock_seconds: float
    stdout_bytes: int
    stderr_preview: str


def cli_invoke(
    cli: str,
    prompt: str,
    model: Optional[str] = None,
    timeout: int = 180,
    cwd: Optional[str] = None,
) -> CLIResponse:
    """Run a one-shot non-interactive CLI invocation.

    Args:
        cli: one of "claude", "codex", "gemini".
        prompt: full prompt text.
        model: optional model override (e.g. "claude-sonnet-4-20250514", "gpt-5",
            "gemini-2.5-pro"). If None, the CLI's configured default is used.
        timeout: subprocess timeout in seconds.
        cwd: working directory for the subprocess. Defaults to current dir.
            Codex requires --skip-git-repo-check when run outside a git tree.

    Returns:
        CLIResponse with the response text and a small trace.

    Raises:
        RuntimeError if the CLI exits non-zero.
        subprocess.TimeoutExpired if it takes longer than `timeout`.
    """
    cli = cli.lower()
    started = time.time()

    if cli == "claude":
        # Claude Code in non-interactive mode. -p triggers print mode; prompt
        # is read from stdin so we don't blow argv limits on long candidate
        # lists. We deliberately do NOT pass --bare: --bare bypasses the
        # OAuth keychain and only accepts ANTHROPIC_API_KEY, defeating the
        # whole point of using the subscription.
        args = ["claude", "-p"]
        if model:
            args.extend(["--model", model])
        proc = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    elif cli == "codex":
        # Codex CLI: `codex exec` is the non-interactive subcommand. Per
        # Roundtable's codex_cli.py, prompt MUST go through stdin (codex
        # >= 0.118 hangs if prompt is on argv with stdin closed).
        args = ["codex", "exec", "--skip-git-repo-check", "--cd", cwd or os.getcwd()]
        if model:
            args.extend(["-m", model])
        proc = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    elif cli == "gemini":
        # Gemini CLI: -p takes the prompt as flag value. For long prompts we
        # could also pipe via stdin, but -p with a positional prompt works
        # for typical sizes here (<128KB).
        args = ["gemini", "-p", prompt]
        if model:
            args.extend(["-m", model])
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    else:
        raise ValueError(f"Unknown CLI {cli!r}; use one of claude / codex / gemini")

    elapsed = round(time.time() - started, 3)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cli} exited with code {proc.returncode}\n"
            f"stderr: {proc.stderr[:500]}\n"
            f"stdout: {proc.stdout[:500]}"
        )

    text = _extract_response_text(cli, proc.stdout)
    return CLIResponse(
        text=text,
        cli=cli,
        model=model,
        returncode=proc.returncode,
        wall_clock_seconds=elapsed,
        stdout_bytes=len(proc.stdout),
        stderr_preview=proc.stderr[:200] if proc.stderr else "",
    )


def _extract_response_text(cli: str, stdout: str) -> str:
    """Return the assistant's response text from each CLI's stdout shape.

    - claude --bare -p: stdout is the response text directly.
    - codex exec: stdout is plain assistant text (without --json).
    - gemini -p: stdout is the response text directly.
    """
    return stdout.strip()


T = TypeVar("T")


def cli_invoke_many(
    items: List[T],
    invoker: Callable[[T], CLIResponse],
    max_concurrency: int = 5,
) -> List[CLIResponse]:
    """Run `invoker(item)` for each item with bounded thread concurrency.

    Preserves input order. Each invoker call is independent — typically a
    closure around `cli_invoke` with its own prompt. Failures propagate as
    exceptions of the corresponding result slot.
    """
    if max_concurrency <= 1:
        return [invoker(item) for item in items]
    results: List[Optional[CLIResponse]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_concurrency) as ex:
        futures = {ex.submit(invoker, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
    return [r for r in results if r is not None]


def parse_json_response(text: str) -> Optional[object]:
    """Best-effort JSON parse, handling fenced code blocks. Returns None on failure."""
    candidates = [text]
    if "```" in text:
        # try stripping a fenced code block
        parts = text.split("```")
        for chunk in parts:
            chunk = chunk.strip()
            if chunk.startswith("json\n"):
                chunk = chunk[5:]
            candidates.append(chunk)
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None
