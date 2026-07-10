#!/usr/bin/env python3
r"""Update all markdown files that use markdown-code-runner for auto-generation.

Run from repo root: python docs/run_markdown_code_runner.py

This wrapper drives the ``markdown-code-runner`` library in-process (rather than
shelling out to its CLI) so it behaves correctly on Windows:

- Files are always read and written as UTF-8 with LF (``\n``) line endings.
  The library's own CLI opens files in the platform default encoding/newline
  mode, which corrupts emoji on Windows (cp1252) and rewrites every file with
  CRLF, producing huge spurious diffs against this LF repo.
- ``CODE:BASH`` blocks are executed through a real POSIX shell. The library
  runs bash blocks with ``subprocess.run(..., shell=True)``, which on Windows
  is ``cmd.exe`` -- it can't parse ``export`` or single quotes, so the embedded
  ``agent-cli ... --help`` output is silently lost. We reroute those calls
  through ``bash -c`` on Windows.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import markdown_code_runner as mcr
from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable

if sys.platform == "win32":
    # When stdout/stderr are piped or redirected on Windows, Python falls back to
    # the legacy ANSI codepage (cp1252), which can't encode the ✓/✗ status glyphs
    # and crashes. Force UTF-8 before building the Rich Console.
    for _stream in (sys.stdout, sys.stderr):
        if isinstance(_stream, io.TextIOWrapper):
            with suppress(ValueError):
                _stream.reconfigure(encoding="utf-8", errors="replace")

console = Console()

# Fixed terminal width for reproducible Rich output in CLI --help commands
FIXED_TERMINAL_WIDTH = "90"


def _find_bash() -> str | None:
    """Locate a POSIX bash on Windows (needed for CODE:BASH blocks)."""
    found = shutil.which("bash")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


class _BashShell:
    """Proxy for the ``subprocess`` module that runs shell commands via bash.

    The markdown-code-runner library calls ``subprocess.run(cmd, shell=True)``
    for bash blocks. On Windows ``shell=True`` uses ``cmd.exe``; we intercept
    those string+shell calls and run them under bash instead. Everything else
    is delegated to the real ``subprocess`` module unchanged.
    """

    def __init__(self, bash: str) -> None:
        self._bash = bash

    def __getattr__(self, name: str) -> object:
        return getattr(subprocess, name)

    def run(self, *args: object, **kwargs: object) -> subprocess.CompletedProcess:
        if kwargs.get("shell") and args and isinstance(args[0], str):
            command = args[0]
            # Force UTF-8 decoding: agent-cli emits UTF-8 (box-drawing, bullets),
            # but subprocess text mode would decode it with the Windows cp1252
            # codepage and crash. encoding=... implies text mode.
            kwargs = {**kwargs, "shell": False, "encoding": "utf-8", "errors": "replace"}
            kwargs.pop("text", None)
            return subprocess.run(  # noqa: PLW1510
                [self._bash, "-c", command],
                *args[1:],  # type: ignore[arg-type]
                **kwargs,  # type: ignore[arg-type]
            )
        return subprocess.run(*args, **kwargs)  # type: ignore[call-overload] # noqa: PLW1510


def _make_include_section(base_path: Path | None) -> Callable[..., str]:
    """UTF-8 replacement for the library's include_section factory.

    Identical to ``markdown_code_runner._create_include_section_func`` except the
    included file is read as UTF-8. The library reads it with the platform default
    codepage, which crashes on emoji (e.g. README.md) on Windows.
    """

    def include_section(file: str, name: str, *, strip_heading: bool = False) -> str:
        path = Path(file)
        if not path.is_absolute() and base_path is not None:
            path = base_path.parent / path
        content = path.read_text(encoding="utf-8")

        start_marker = f"<!-- SECTION:{name}:START -->"
        end_marker = f"<!-- SECTION:{name}:END -->"
        start_idx = content.find(start_marker)
        if start_idx == -1:
            msg = f"Section '{name}' not found in {file}"
            raise ValueError(msg)
        end_idx = content.find(end_marker, start_idx)
        if end_idx == -1:
            msg = f"End marker for section '{name}' not found in {file}"
            raise ValueError(msg)
        result = content[start_idx + len(start_marker) : end_idx].strip()
        if strip_heading:
            result = re.sub(r"^#{1,6}\s+[^\n]+\n+", "", result, count=1)
        return result

    return include_section


def find_markdown_files_with_code_blocks(docs_dir: Path) -> list[Path]:
    """Find all markdown files containing markdown-code-runner markers."""
    files_with_code = []
    for md_file in docs_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        # Match both CODE:START and CODE:BASH:START patterns
        if "<!-- CODE:START -->" in content or "<!-- CODE:BASH:START -->" in content:
            files_with_code.append(md_file)
    return sorted(files_with_code)


def _update_markdown_file(path: Path) -> None:
    """Regenerate one file in-process, always writing UTF-8 with LF endings.

    Mirrors ``markdown_code_runner.update_markdown_file`` but pins the I/O
    encoding and newline so the output is byte-for-byte what CI (Linux) produces.
    ``splitlines`` also normalizes any pre-existing CRLF back to LF.
    """
    original_lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = mcr.process_markdown(original_lines, base_path=path)
    updated = "\n".join(new_lines).rstrip() + "\n"
    path.write_text(updated, encoding="utf-8", newline="\n")


def run_markdown_code_runner(files: list[Path], repo_root: Path) -> bool:
    """Run markdown-code-runner on all files. Returns True if all succeeded."""
    if not files:
        console.print("No files with CODE:START markers found.")
        return True

    # include_section() must read included files as UTF-8 (README has emoji).
    mcr._create_include_section_func = _make_include_section  # type: ignore[attr-defined]

    # On Windows, bash blocks must run under bash, not cmd.exe. Route the
    # library's shell calls through bash for the duration of this run.
    if sys.platform == "win32":
        bash = _find_bash()
        if bash is None:
            console.print(
                "[red]bash not found.[/red] CODE:BASH blocks need a POSIX shell "
                "(install Git for Windows, which provides bash).",
            )
            return False
        mcr.subprocess = _BashShell(bash)  # type: ignore[attr-defined]

    console.print(f"Found {len(files)} file(s) with auto-generated content:")
    for f in files:
        console.print(f"  - {f.relative_to(repo_root)}")
    console.print()

    all_success = True
    for file in files:
        rel_path = file.relative_to(repo_root)
        console.print(f"Updating {rel_path}...", end=" ")
        try:
            _update_markdown_file(file)
        except Exception as exc:  # report and keep going
            console.print("[red]✗[/red]")
            console.print(f"  [red]Error:[/red] {exc}")
            all_success = False
        else:
            console.print("[green]✓[/green]")

    return all_success


def main() -> int:
    """Main entry point."""
    repo_root = Path(__file__).parent.parent

    # Set fixed terminal width for reproducible Rich/Typer CLI help output
    os.environ["COLUMNS"] = FIXED_TERMINAL_WIDTH  # Rich Console width
    os.environ["TERMINAL_WIDTH"] = FIXED_TERMINAL_WIDTH  # Typer MAX_WIDTH for help panels
    # Prevent Typer from forcing terminal mode in CI (GITHUB_ACTIONS),
    # which treats TERM=dumb as a fixed 80-column terminal.
    os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"

    files = find_markdown_files_with_code_blocks(repo_root)
    success = run_markdown_code_runner(files, repo_root)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
