"""Tests for the transcribe-live agent."""

from __future__ import annotations

import asyncio
import io
import json
import platform
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_cli import config, constants
from agent_cli.agents.transcribe_live import (
    _DEFAULT_AUDIO_DIR,
    _DEFAULT_LOG_FILE,
    _MIN_SEGMENT_DURATION_SECONDS,
    _MSG_BYE,
    _MSG_LISTENING,
    _MSG_PROCESSING,
    _MSG_READY,
    _TRANSCRIPT_BEGIN,
    _TRANSCRIPT_END,
    DaemonConfig,
    _generate_audio_path,
    _log_segment,
    _next_stdio_action,
    _process_segment,
    _segment_duration_seconds,
    _stdio_daemon_loop,
    transcribe_live,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def temp_log_file(tmp_path: Path) -> Path:
    """Create a temporary log file path."""
    return tmp_path / "transcriptions.jsonl"


@pytest.fixture
def temp_audio_dir(tmp_path: Path) -> Path:
    """Create a temporary audio directory."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    return audio_dir


def test_log_segment(temp_log_file: Path, tmp_path: Path) -> None:
    """Test logging a transcription segment."""
    audio_file = tmp_path / "test.mp3"
    timestamp = datetime.now(UTC)
    _log_segment(
        temp_log_file,
        timestamp=timestamp,
        role="test",
        raw_output="hello world",
        processed_output="Hello, world.",
        audio_file=audio_file,
        duration_seconds=2.5,
        model_info="test:model",
    )

    # Read and verify log entry
    assert temp_log_file.exists()
    with temp_log_file.open() as f:
        line = f.readline()
        entry = json.loads(line)

    assert entry["role"] == "test"
    assert entry["raw_output"] == "hello world"
    assert entry["processed_output"] == "Hello, world."
    assert entry["audio_file"] == str(audio_file)
    assert entry["duration_seconds"] == 2.5
    assert entry["model"] == "test:model"


def test_log_segment_creates_parent_dirs(tmp_path: Path) -> None:
    """Test that log_segment creates parent directories."""
    log_file = tmp_path / "nested" / "dir" / "log.jsonl"

    _log_segment(
        log_file,
        timestamp=datetime.now(UTC),
        role="test",
        raw_output="test",
        processed_output=None,
        audio_file=None,
        duration_seconds=1.0,
    )

    assert log_file.exists()


def test_generate_audio_path(temp_audio_dir: Path) -> None:
    """Test audio path generation with date-based structure."""
    timestamp = datetime(2025, 1, 15, 10, 30, 45, 123000, tzinfo=UTC)
    path = _generate_audio_path(temp_audio_dir, timestamp)

    assert path.suffix == ".mp3"
    assert path.parts[-4:-1] == ("2025", "01", "15")  # Date directories
    assert "103045" in path.name  # HHMMSS


def test_default_audio_dir() -> None:
    """Test default audio directory path."""
    assert _DEFAULT_AUDIO_DIR.name == "audio"
    assert ".config" in str(_DEFAULT_AUDIO_DIR)
    assert "agent-cli" in str(_DEFAULT_AUDIO_DIR)


def test_default_log_file() -> None:
    """Test default log file path."""
    assert _DEFAULT_LOG_FILE.name == "transcriptions.jsonl"
    assert ".config" in str(_DEFAULT_LOG_FILE)
    assert "agent-cli" in str(_DEFAULT_LOG_FILE)


def test_transcribe_live_command_exists() -> None:
    """Test that the transcribe-live command is registered."""
    assert callable(transcribe_live)


def test_min_segment_duration_constant() -> None:
    """Test that the minimum segment duration constant is defined."""
    assert _MIN_SEGMENT_DURATION_SECONDS == 0.3


@pytest.fixture
def mock_vad() -> MagicMock:
    """Create a mock VoiceActivityDetector."""
    vad = MagicMock()
    vad.get_segment_duration_seconds.return_value = 1.0  # 1 second segment
    return vad


@pytest.fixture
def daemon_config(tmp_path: Path, mock_vad: MagicMock) -> DaemonConfig:
    """Create a DaemonConfig for testing."""
    return DaemonConfig(
        role="test",
        vad=mock_vad,
        input_device_index=0,
        provider=config.ProviderSelection(
            asr_provider="wyoming",
            llm_provider="ollama",
            tts_provider="wyoming",
        ),
        wyoming_asr=config.WyomingASR(
            asr_wyoming_ip="localhost",
            asr_wyoming_port=10300,
        ),
        openai_asr=config.OpenAIASR(
            asr_openai_model="whisper-1",
            openai_api_key=None,
            openai_base_url=None,
            asr_openai_prompt=None,
        ),
        gemini_asr=config.GeminiASR(
            asr_gemini_model="gemini-2.0-flash",
            gemini_api_key=None,
        ),
        ollama=config.Ollama(
            llm_ollama_model="gemma3:4b",
            llm_ollama_host="http://localhost:11434",
        ),
        openai_llm=config.OpenAILLM(
            llm_openai_model="gpt-4",
            openai_api_key=None,
            openai_base_url=None,
        ),
        gemini_llm=config.GeminiLLM(
            llm_gemini_model="gemini-2.0-flash",
            gemini_api_key=None,
        ),
        llm_enabled=False,
        save_audio=False,
        audio_dir=tmp_path / "audio",
        log_file=tmp_path / "transcriptions.jsonl",
        quiet=True,
        clipboard=False,
    )


def test_daemon_config_creation(daemon_config: DaemonConfig) -> None:
    """Test that DaemonConfig can be created with all required fields."""
    assert daemon_config.role == "test"
    assert daemon_config.llm_enabled is False
    assert daemon_config.save_audio is False
    assert daemon_config.quiet is True
    assert daemon_config.clipboard is False


@pytest.mark.asyncio
async def test_process_segment_skips_short_segments(
    daemon_config: DaemonConfig,
) -> None:
    """Test that _process_segment skips segments shorter than minimum duration."""
    # Duration is derived from the byte length: 1000 bytes = 500 samples ≈ 0.03s,
    # which is below _MIN_SEGMENT_DURATION_SECONDS (0.3s).
    timestamp = datetime.now(UTC)
    segment = b"\x00" * 1000

    # Should return early without processing
    with patch(
        "agent_cli.agents.transcribe_live.create_recorded_audio_transcriber",
    ) as mock_transcriber:
        await _process_segment(daemon_config, segment, timestamp)
        # Transcriber should not be called for short segments
        mock_transcriber.assert_not_called()


@pytest.mark.asyncio
async def test_process_segment_transcribes_audio(
    daemon_config: DaemonConfig,
) -> None:
    """Test that _process_segment transcribes audio and logs result."""
    timestamp = datetime.now(UTC)
    segment = b"\x00" * 32000  # 1 second of audio

    mock_transcriber = AsyncMock(return_value="Hello world")

    with patch(
        "agent_cli.agents.transcribe_live.create_recorded_audio_transcriber",
        return_value=mock_transcriber,
    ):
        await _process_segment(daemon_config, segment, timestamp)

    # Verify transcription was called
    mock_transcriber.assert_called_once()

    # Verify log file was written
    assert daemon_config.log_file.exists()
    with daemon_config.log_file.open() as f:
        entry = json.loads(f.readline())
    assert entry["raw_output"] == "Hello world"
    assert entry["role"] == "test"


@pytest.mark.asyncio
async def test_process_segment_skips_empty_transcript(
    daemon_config: DaemonConfig,
) -> None:
    """Test that _process_segment skips logging for empty transcripts."""
    timestamp = datetime.now(UTC)
    segment = b"\x00" * 32000

    mock_transcriber = AsyncMock(return_value="")  # Empty transcript

    with patch(
        "agent_cli.agents.transcribe_live.create_recorded_audio_transcriber",
        return_value=mock_transcriber,
    ):
        await _process_segment(daemon_config, segment, timestamp)

    # Log file should not exist (no log entry written)
    assert not daemon_config.log_file.exists()


@pytest.mark.asyncio
async def test_process_segment_with_llm_enabled(
    daemon_config: DaemonConfig,
) -> None:
    """Test that _process_segment uses LLM when enabled."""
    daemon_config.llm_enabled = True
    timestamp = datetime.now(UTC)
    segment = b"\x00" * 32000

    mock_transcriber = AsyncMock(return_value="hello world")
    mock_llm_processor = AsyncMock(return_value="Hello, world.")

    with (
        patch(
            "agent_cli.agents.transcribe_live.create_recorded_audio_transcriber",
            return_value=mock_transcriber,
        ),
        patch(
            "agent_cli.agents.transcribe_live.process_and_update_clipboard",
            mock_llm_processor,
        ),
    ):
        await _process_segment(daemon_config, segment, timestamp)

    # LLM processor should be called
    mock_llm_processor.assert_called_once()

    # Log should contain both raw and processed output
    with daemon_config.log_file.open() as f:
        entry = json.loads(f.readline())
    assert entry["raw_output"] == "hello world"
    assert entry["processed_output"] == "Hello, world."
    assert "ollama" in entry["model"]


@pytest.mark.asyncio
async def test_process_segment_with_clipboard(
    daemon_config: DaemonConfig,
) -> None:
    """Test that _process_segment copies to clipboard when enabled."""
    daemon_config.clipboard = True
    timestamp = datetime.now(UTC)
    segment = b"\x00" * 32000

    mock_transcriber = AsyncMock(return_value="Hello world")

    with (
        patch(
            "agent_cli.agents.transcribe_live.create_recorded_audio_transcriber",
            return_value=mock_transcriber,
        ),
        patch("pyperclip.copy") as mock_copy,
    ):
        await _process_segment(daemon_config, segment, timestamp)

    # Clipboard should be updated
    mock_copy.assert_called_once_with("Hello world")


@pytest.mark.asyncio
async def test_process_segment_saves_audio(
    daemon_config: DaemonConfig,
) -> None:
    """Test that _process_segment saves audio as MP3 when enabled."""
    daemon_config.save_audio = True
    timestamp = datetime.now(UTC)
    segment = b"\x00" * 32000

    mock_transcriber = AsyncMock(return_value="Hello world")

    with (
        patch(
            "agent_cli.agents.transcribe_live.create_recorded_audio_transcriber",
            return_value=mock_transcriber,
        ),
        patch(
            "agent_cli.agents.transcribe_live.save_audio_as_mp3",
        ) as mock_save_mp3,
    ):
        await _process_segment(daemon_config, segment, timestamp)

    # MP3 save should be called
    mock_save_mp3.assert_called_once()
    call_args = mock_save_mp3.call_args
    assert call_args[0][0] == segment  # First arg is segment
    assert call_args[0][1].suffix == ".mp3"  # Second arg is path with .mp3 extension


@pytest.mark.asyncio
async def test_process_segment_handles_mp3_save_error(
    daemon_config: DaemonConfig,
) -> None:
    """Test that _process_segment handles MP3 save errors gracefully."""
    daemon_config.save_audio = True
    timestamp = datetime.now(UTC)
    segment = b"\x00" * 32000

    mock_transcriber = AsyncMock(return_value="Hello world")

    with (
        patch(
            "agent_cli.agents.transcribe_live.create_recorded_audio_transcriber",
            return_value=mock_transcriber,
        ),
        patch(
            "agent_cli.agents.transcribe_live.save_audio_as_mp3",
            side_effect=RuntimeError("FFmpeg not found"),
        ),
    ):
        # Should not raise, just log the error
        await _process_segment(daemon_config, segment, timestamp)

    # Transcription should still be logged
    assert daemon_config.log_file.exists()


@pytest.mark.asyncio
async def test_process_segment_with_openai_provider(
    daemon_config: DaemonConfig,
) -> None:
    """Test that _process_segment uses correct transcriber for OpenAI provider."""
    daemon_config.provider.asr_provider = "openai"
    timestamp = datetime.now(UTC)
    segment = b"\x00" * 32000

    mock_transcriber = AsyncMock(return_value="Hello world")

    with patch(
        "agent_cli.agents.transcribe_live.create_recorded_audio_transcriber",
        return_value=mock_transcriber,
    ):
        await _process_segment(daemon_config, segment, timestamp)

    # Verify transcriber was called with correct arguments for OpenAI
    mock_transcriber.assert_called_once()
    # OpenAI provider passes positional args
    call_args = mock_transcriber.call_args
    assert call_args[0][0] == segment

    # Log should contain openai model info
    with daemon_config.log_file.open() as f:
        entry = json.loads(f.readline())
    assert "openai" in entry["model"]


def test_log_segment_includes_hostname(temp_log_file: Path) -> None:
    """Test that log entries include hostname."""
    _log_segment(
        temp_log_file,
        timestamp=datetime.now(UTC),
        role="test",
        raw_output="test",
        processed_output=None,
        audio_file=None,
        duration_seconds=1.0,
    )

    with temp_log_file.open() as f:
        entry = json.loads(f.readline())

    assert entry["hostname"] == platform.node()


def test_log_segment_handles_unicode(temp_log_file: Path) -> None:
    """Test that log entries handle unicode characters correctly."""
    _log_segment(
        temp_log_file,
        timestamp=datetime.now(UTC),
        role="test",
        raw_output="Hello 世界 🌍",
        processed_output="Привет мир",
        audio_file=None,
        duration_seconds=1.0,
    )

    with temp_log_file.open(encoding="utf-8") as f:
        entry = json.loads(f.readline())

    assert entry["raw_output"] == "Hello 世界 🌍"
    assert entry["processed_output"] == "Привет мир"


def test_generate_audio_path_creates_directories(tmp_path: Path) -> None:
    """Test that audio path generation creates necessary directories."""
    audio_dir = tmp_path / "audio"
    # Directory doesn't exist yet
    assert not audio_dir.exists()

    timestamp = datetime(2025, 6, 15, 14, 30, 0, tzinfo=UTC)
    path = _generate_audio_path(audio_dir, timestamp)

    # Directory should now exist
    assert path.parent.exists()
    assert path.parent.name == "15"  # Day
    assert path.parent.parent.name == "06"  # Month
    assert path.parent.parent.parent.name == "2025"  # Year


def test_generate_audio_path_includes_milliseconds(tmp_path: Path) -> None:
    """Test that audio path includes milliseconds for uniqueness."""
    audio_dir = tmp_path / "audio"
    timestamp = datetime(2025, 1, 15, 10, 30, 45, 567000, tzinfo=UTC)
    path = _generate_audio_path(audio_dir, timestamp)

    # Should include milliseconds in filename
    assert "567" in path.name


# --- stdio push-to-talk mode ---


def test_segment_duration_seconds() -> None:
    """Duration is derived from byte length, sample width, and rate."""
    # 1 second of 16-bit mono audio = AUDIO_RATE samples * 2 bytes.
    one_second = b"\x00" * (constants.AUDIO_RATE * constants.AUDIO_FORMAT_WIDTH)
    assert _segment_duration_seconds(one_second) == 1.0
    assert _segment_duration_seconds(b"") == 0.0


@pytest.mark.parametrize(
    ("command", "listening", "expected"),
    [
        # start / toggle from idle begins listening
        ("start", False, ("start", True)),
        ("toggle", False, ("start", True)),
        # stop / toggle while listening finalizes
        ("stop", True, ("stop", False)),
        ("toggle", True, ("stop", False)),
        # quit is honored in either state, preserving current listening flag
        ("quit", False, ("quit", False)),
        ("quit", True, ("quit", True)),
        ("exit", False, ("quit", False)),
        # redundant transitions keep state without action
        ("start", True, ("noop", True)),
        ("stop", False, ("noop", False)),
        # anything else is unknown
        ("frobnicate", False, ("unknown", False)),
    ],
)
def test_next_stdio_action(
    command: str,
    listening: bool,
    expected: tuple[str, bool],
) -> None:
    """The command state machine maps (command, state) to (action, new state)."""
    assert _next_stdio_action(command, listening=listening) == expected


@pytest.mark.asyncio
async def test_process_segment_returns_final_text(
    daemon_config: DaemonConfig,
) -> None:
    """_process_segment returns the raw transcript when the LLM is disabled."""
    segment = b"\x00" * 32000  # 1 second

    mock_transcriber = AsyncMock(return_value="Hello world")
    with patch(
        "agent_cli.agents.transcribe_live.create_recorded_audio_transcriber",
        return_value=mock_transcriber,
    ):
        result = await _process_segment(daemon_config, segment, datetime.now(UTC))

    assert result == "Hello world"


@pytest.mark.asyncio
async def test_process_segment_returns_none_for_short_segment(
    daemon_config: DaemonConfig,
) -> None:
    """Segments below the minimum duration return None (nothing transcribed)."""
    segment = b"\x00" * 100  # far below _MIN_SEGMENT_DURATION_SECONDS
    result = await _process_segment(daemon_config, segment, datetime.now(UTC))
    assert result is None


def _fake_audio_stream() -> MagicMock:
    """A stand-in sounddevice stream whose read() yields silent PCM chunks."""
    fake_data = MagicMock()
    fake_data.tobytes.return_value = b"\x00" * constants.AUDIO_CHUNK_SIZE
    stream = MagicMock()
    stream.read.return_value = (fake_data, None)
    context = MagicMock()
    context.__enter__.return_value = stream
    context.__exit__.return_value = False
    return context


@pytest.mark.asyncio
async def test_stdio_daemon_loop_start_stop_cycle(
    daemon_config: DaemonConfig,
) -> None:
    """A start/stop cycle over stdin emits the protocol events and the transcript."""
    daemon_config.vad = None
    commands = io.StringIO("start\nstop\n")  # EOF after stop triggers quit
    out = io.StringIO()
    fake_ctx = _fake_audio_stream()

    with (
        patch("agent_cli.agents.transcribe_live.setup_input_stream", return_value=MagicMock()),
        patch(
            "agent_cli.agents.transcribe_live.open_audio_stream",
            return_value=fake_ctx,
        ),
        patch(
            "agent_cli.agents.transcribe_live._process_segment",
            AsyncMock(return_value="hello world"),
        ),
        patch("sys.stdin", commands),
        patch("sys.stdout", out),
    ):
        await asyncio.wait_for(_stdio_daemon_loop(daemon_config), timeout=5.0)

    lines = [line for line in out.getvalue().splitlines() if line]
    assert lines[0] == _MSG_READY
    assert _MSG_LISTENING in lines
    assert _MSG_PROCESSING in lines
    assert lines[-1] == _MSG_BYE
    # Transcript is delivered between the sentinels.
    begin = lines.index(_TRANSCRIPT_BEGIN)
    end = lines.index(_TRANSCRIPT_END)
    assert lines[begin + 1 : end] == ["hello world"]
    # The mic is opened on start and released on stop.
    fake_ctx.__enter__.assert_called_once()
    fake_ctx.__exit__.assert_called_once()


@pytest.mark.asyncio
async def test_stdio_daemon_loop_unknown_command(
    daemon_config: DaemonConfig,
) -> None:
    """Unknown commands are reported without stopping the daemon."""
    daemon_config.vad = None
    commands = io.StringIO("wat\nquit\n")
    out = io.StringIO()
    fake_ctx = _fake_audio_stream()

    with (
        patch("agent_cli.agents.transcribe_live.setup_input_stream", return_value=MagicMock()),
        patch(
            "agent_cli.agents.transcribe_live.open_audio_stream",
            return_value=fake_ctx,
        ),
        patch("sys.stdin", commands),
        patch("sys.stdout", out),
    ):
        await asyncio.wait_for(_stdio_daemon_loop(daemon_config), timeout=5.0)

    output = out.getvalue()
    assert "ERROR unknown command: wat" in output
    assert output.strip().endswith(_MSG_BYE)
    # Without a start command the microphone is never opened.
    fake_ctx.__enter__.assert_not_called()
