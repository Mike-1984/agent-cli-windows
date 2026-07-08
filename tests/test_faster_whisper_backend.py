"""Tests for the faster-whisper backend."""

from __future__ import annotations

import os
import sys
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, patch

import pytest

from agent_cli.server.whisper.backends.base import BackendConfig
from agent_cli.server.whisper.backends.faster_whisper import (
    FasterWhisperBackend,
    _prepend_nvidia_dll_dirs,
    _resolve_device,
)

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path


def test_resolve_device_auto_falls_back_to_cpu_without_cuda_libs() -> None:
    """Fall back to CPU when device is `auto` and CUDA libraries cannot load.

    Evidence: the faster-whisper README states GPU execution requires cuBLAS
    for CUDA 12 and cuDNN 9 (https://github.com/SYSTRAN/faster-whisper#gpu);
    without them CTranslate2 raises `RuntimeError: Library cublas64_12.dll is
    not found or cannot be loaded` at model load/inference time, so `auto`
    must not resolve to CUDA unless those libraries are actually loadable.
    """
    with (
        patch(
            "agent_cli.server.whisper.backends.faster_whisper._cuda_libs_available",
            return_value=False,
        ),
        patch(
            "agent_cli.server.whisper.backends.faster_whisper._prepend_nvidia_dll_dirs",
        ),
    ):
        assert _resolve_device("auto") == "cpu"


def test_resolve_device_respects_explicit_cuda_without_cuda_libs() -> None:
    """Keep an explicitly requested CUDA device even if the check fails.

    The availability check may have false negatives (e.g. libraries in a
    non-standard location), so an explicit `--device cuda` is only warned
    about, never silently rewritten to CPU.
    """
    with (
        patch(
            "agent_cli.server.whisper.backends.faster_whisper._cuda_libs_available",
            return_value=False,
        ),
        patch(
            "agent_cli.server.whisper.backends.faster_whisper._prepend_nvidia_dll_dirs",
        ),
    ):
        assert _resolve_device("cuda") == "cuda"
        assert _resolve_device("cuda:0") == "cuda:0"


def test_resolve_device_passes_cpu_through_without_checks() -> None:
    """`cpu` (and other non-CUDA devices) skip the CUDA library check."""
    with patch(
        "agent_cli.server.whisper.backends.faster_whisper._cuda_libs_available",
    ) as check:
        assert _resolve_device("cpu") == "cpu"
        check.assert_not_called()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL search path behavior")
def test_prepend_nvidia_dll_dirs_adds_wheel_bin_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepend site-packages/nvidia/*/bin to PATH so CTranslate2 finds the DLLs.

    Evidence: the nvidia-cublas-cu12 and nvidia-cudnn-cu12 wheels install
    their DLLs into site-packages/nvidia/{cublas,cudnn}/bin (verifiable via
    `pip show -f nvidia-cublas-cu12`), which is not on the Windows DLL search
    path; CTranslate2 loads cublas64_12.dll via LoadLibrary, which searches
    PATH and directories registered with AddDllDirectory.
    """
    bin_dir = tmp_path / "nvidia" / "cublas" / "bin"
    bin_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "sysconfig.get_paths",
        lambda: {"purelib": str(tmp_path)},
    )
    monkeypatch.setenv("PATH", "existing")

    _prepend_nvidia_dll_dirs()

    assert os.environ["PATH"] == f"{bin_dir}{os.pathsep}existing"


@pytest.mark.asyncio
async def test_faster_whisper_transcribe_recovers_from_broken_process_pool() -> None:
    """Reload the backend and retry once when the process pool is broken."""
    config = BackendConfig(model_name="tiny", device="cpu", compute_type="int8")
    backend = FasterWhisperBackend(config)
    initial_executor = cast("ProcessPoolExecutor", object())
    backend._executor = initial_executor
    backend._device = "cpu"

    recovered_executor = cast("ProcessPoolExecutor", object())
    fake_result = {
        "text": "hello world",
        "language": "en",
        "language_probability": 0.99,
        "duration": 1.25,
        "segments": [],
    }
    executors_seen: list[object] = []

    async def mock_run_in_executor(
        executor: object, _func: object, *_args: object
    ) -> dict[str, object]:
        executors_seen.append(executor)
        if len(executors_seen) == 1:
            msg = "worker died"
            raise BrokenProcessPool(msg)
        return fake_result

    async def fake_unload() -> None:
        backend._executor = None
        backend._device = None

    async def fake_load() -> float:
        backend._executor = recovered_executor
        backend._device = "cpu"
        return 0.1

    with (
        patch("asyncio.get_running_loop") as mock_loop,
        patch.object(backend, "unload", new=AsyncMock(side_effect=fake_unload)) as unload_mock,
        patch.object(backend, "load", new=AsyncMock(side_effect=fake_load)) as load_mock,
    ):
        mock_loop.return_value.run_in_executor = mock_run_in_executor
        result = await backend.transcribe(b"fake audio bytes")

    assert result.text == "hello world"
    unload_mock.assert_awaited_once()
    load_mock.assert_awaited_once()
    assert executors_seen == [initial_executor, recovered_executor]
