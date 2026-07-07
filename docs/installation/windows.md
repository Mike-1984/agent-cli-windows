---
icon: fontawesome/brands/windows
---

# Windows Installation Guide

> [!NOTE]
> **Verified on real Windows hardware.** The steps below (native Ollama + faster-whisper +
> Piper, no Docker, no WSL) were tested end-to-end on Windows 11 (laptop with an NVIDIA
> T1200, 4GB VRAM): `autocorrect` through Ollama, `speak` through Piper, and ASR through the
> Whisper HTTP endpoint all produced correct output. Three real bugs turned up during that
> testing and are fixed in this fork (see [Known Windows Fixes](#known-windows-fixes-in-this-fork)
> below) — if you're installing from upstream `agent-cli` or from PyPI instead of this fork,
> you may still hit them.
>
> Not independently verified in this pass: microphone-driven `transcribe` (no mic in the test
> environment), the AutoHotkey hotkeys below, GPU/CUDA Whisper (skipped — 4GB VRAM is too
> tight for anything but `tiny`/`base`), and the `setup-windows.ps1` / `start-all-services-windows.ps1`
> scripts (the manual steps below were run instead, and are equivalent).

`agent-cli` works natively on Windows - no WSL required! All services (Ollama, Whisper, Piper) run directly on Windows.

> [!TIP]
> Have Docker Desktop already? The `docker/docker-compose.yml` services (Whisper, Piper,
> Ollama, RAG/memory proxies) are Linux images — they need Docker Desktop in **Linux
> containers** mode. If you use Docker Desktop for Windows containers (e.g. for other
> projects), switching modes is disruptive and this native path below avoids Docker
> entirely.

## Prerequisites

- Windows 10/11
- 8GB+ RAM (16GB+ recommended for GPU acceleration)
- 10GB free disk space

### For GPU Acceleration (Optional)

- NVIDIA GPU (GTX 1060+ or RTX series recommended; 4GB VRAM cards like laptop T1200s can only
  fit small Whisper models such as `tiny`/`base` — CPU with `small` is often more reliable)
- NVIDIA drivers installed
- CUDA 12 and cuDNN 9 (see [faster-whisper GPU docs](https://github.com/SYSTRAN/faster-whisper#gpu))

## Quick Start (Cloud Providers)

The fastest way to get started - no local services needed:

```powershell
# Install uv (Python package manager)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install agent-cli
uv tool install agent-cli

# Use with cloud providers (requires API keys)
$env:OPENAI_API_KEY = "sk-..."
agent-cli transcribe --asr-provider openai --llm-provider openai
```

---

## Full Local Setup (Recommended)

For a completely local setup with no internet dependency. This is the exact sequence that was
verified end-to-end.

1. **Install uv:**

   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Install Ollama** (via winget, or download from [ollama.com](https://ollama.com/download/windows)):

   ```powershell
   winget install --id Ollama.Ollama -e
   ollama pull gemma3:4b
   ```

   Ollama registers itself as a background service on install, so `ollama serve` is usually
   already running — check with `ollama list`.

3. **Install agent-cli with the extras you need.** `audio` and `llm` give you the CLI's mic/LLM
   commands; `faster-whisper` and `piper` give you the local ASR/TTS servers:

   ```powershell
   uv tool install "agent-cli[audio,llm,faster-whisper,piper]"
   ```

   To install this fork's Windows fixes instead of the last PyPI release, clone and install
   editable:

   ```powershell
   git clone https://github.com/Mike-1984/agent-cli-windows.git
   cd agent-cli-windows
   uv tool install --editable ".[audio,llm,faster-whisper,piper]"
   ```

4. **Run the ASR and TTS servers** (each in its own terminal — leave them running):

   ```powershell
   # Whisper ASR - "small" on CPU is a safe default for laptops without a beefy GPU
   agent-cli server whisper --model small --device cpu

   # Piper TTS
   agent-cli server tts --backend piper
   ```

   `agent-cli` also auto-installs missing extras the first time a command needs them (see
   `agent_cli/core/deps.py`), so `uv tool install --upgrade agent-cli` with no extras, followed
   by `scripts/setup-windows.ps1` / `scripts/start-all-services-windows.ps1`, should also work —
   that path just wasn't the one exercised in this verification pass.

5. **Test it:**

   ```powershell
   agent-cli autocorrect "this has an eror" --llm-provider ollama
   agent-cli speak "hello from windows" --save-file test.wav
   agent-cli transcribe
   ```

---

## Services Overview

| Service     | Port  | GPU Support | Description              |
| ----------- | ----- | ----------- | ------------------------ |
| **Ollama**  | 11434 | ✅ CUDA     | LLM inference            |
| **Whisper** | 10300 | ✅ CUDA     | Speech-to-text (ASR)     |
| **Piper**   | 10200 | N/A         | Text-to-speech (TTS)     |

## GPU Acceleration

- **With a capable GPU (8GB+ VRAM):** `large-v3` gives the best accuracy.
- **With a small/laptop GPU (e.g. 4GB) or CPU-only:** use `tiny`, `base`, or `small` with
  `--device cpu` — `large-v3` will not fit in 4GB VRAM and CPU inference of it is impractical.

To verify GPU is being used:

```powershell
nvidia-smi
```

---

## Global Hotkeys with AutoHotkey

Use [AutoHotkey v2](https://www.autohotkey.com/) for global keyboard shortcuts.

1. Create a file named `agent-cli.ahk`:

```autohotkey
#Requires AutoHotkey v2.0
Persistent

; Win+Shift+W - Toggle transcription
#+w::{
    statusFile := A_Temp . "\agent-cli-status.txt"
    cmd := Format('{1} /C agent-cli transcribe --status > "{2}" 2>&1', A_ComSpec, statusFile)
    RunWait(cmd, , "Hide")
    status := FileRead(statusFile)
    if InStr(status, "not running") {
        TrayTip("Starting transcription...", "agent-cli", 1)
        Run("agent-cli transcribe --toggle", , "Hide")
    } else {
        TrayTip("Stopping transcription...", "agent-cli", 1)
        Run("agent-cli transcribe --toggle", , "Hide")
    }
}

; Win+Shift+A - Autocorrect clipboard
#+a::{
    TrayTip("Autocorrecting...", "agent-cli", 1)
    Run("agent-cli autocorrect", , "Hide")
}

; Win+Shift+E - Voice edit selection
#+e::{
    Send("^c")
    ClipWait(1)
    TrayTip("Voice editing...", "agent-cli", 1)
    Run("agent-cli voice-edit", , "Hide")
}
```

2. Double-click the script to run it.

> [!TIP]
> To run at startup: Press `Win+R`, type `shell:startup`, and place a shortcut to your `.ahk` file there.

> [!NOTE]
> `Run(..., "Hide")` redirects `agent-cli`'s output away from a real console, which used to
> crash on the emoji in its output (see [Known Windows Fixes](#known-windows-fixes-in-this-fork)) —
> fixed in this fork, but a factor to know about if you're on an older build.

---

## Troubleshooting

### Audio device not found

Run `agent-cli transcribe --list-devices` and use `--input-device-index` with your microphone's index.

### Wyoming server connection refused

Ensure the services are running:

```powershell
# Check if ports are in use
netstat -an | findstr "10300 10200 11434"
```

### GPU not being used

1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA installation
3. Set device explicitly: `$env:WHISPER_DEVICE = "cuda"`

### Ollama not responding

Check if Ollama is running:

```powershell
ollama list
```

If not, start it: `ollama serve` or launch from Start Menu.

---

## Known Windows Fixes in This Fork

Found and fixed while verifying the setup above — worth knowing about if you're comparing
against upstream `agent-cli` or an older install:

- **Crash printing emoji output.** When `agent-cli`'s stdout/stderr aren't attached to a real
  console — piped, redirected to a file, or launched hidden via AutoHotkey's `Run "Hide"` (as
  above) — Python fell back to the legacy `cp1252` codepage, which can't encode emoji like 📋,
  and crashed with `UnicodeEncodeError`. Fixed by forcing UTF-8 on `sys.stdout`/`sys.stderr` on
  `win32`.
- **`WinError 32` deleting temp files.** The server-side audio conversion path
  (`convert_audio_to_wyoming_format`) kept temp file handles open while FFmpeg ran and while
  cleaning up, which Windows disallows (unlike POSIX, it won't delete a file another handle
  still has open). Fixed by writing/reading via plain paths instead of held-open handles.
- **`pydantic-ai-slim` API break.** `OpenAIModel` was renamed to `OpenAIChatModel` upstream in
  `pydantic-ai-slim`; the loose version pin resolved to a version where the old name no longer
  exists, breaking every LLM call (Ollama, OpenAI, RAG) on any platform, not just Windows.
