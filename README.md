# Irodori OpenAI TTS Server

OpenAI Text-to-Speech API compatible server for [Irodori-TTS](https://github.com/Aratako/Irodori-TTS).

This server targets the [Irodori-TTS 500M v3 base model](https://huggingface.co/Aratako/Irodori-TTS-500M-v3). It supports reference-audio voice cloning, OpenAI-style response formats, and automatic long text chunking.

Inference runs on the [stak/Irodori-TTS performance fork](https://github.com/stak/Irodori-TTS), which adds CUDA-graph replay, graph prewarming, LoRA hot-swap, an optional watermark toggle, and other inference-speed optimizations on top of upstream Irodori-TTS. See the fork's [performance notes](https://github.com/stak/Irodori-TTS/blob/main/docs/performance.md) for details and tuning.

Streaming synthesis is not implemented. Requests return one complete audio response.

## Features

- OpenAI-compatible `POST /v1/audio/speech`
- Reference voices from files, `voices.json`, or HTTP upload
- Response formats: `wav`, `mp3`, `flac`, `opus`, `aac`, `pcm`
- Automatic long text chunking
- Per-request dynamic LoRA adapter loading
- CUDA graph prewarming at startup or via `POST /v1/admin/prewarm`
- LoRA hot-swap that keeps prewarmed CUDA graphs across adapter switches
- Optional SilentCipher watermark disable (per request or by default)
- Optional bearer token auth

## Requirements

For local Python:

- Python 3.10
- uv
- FFmpeg for compressed audio formats

For Docker:

- Docker Engine with Docker Compose, or Docker Desktop
- NVIDIA Container Toolkit or Docker Desktop GPU support for CUDA inference
- ROCm-capable Docker host for AMD GPU inference

A CUDA or ROCm GPU is recommended for practical inference.

## Installation

```bash
git clone https://github.com/Aratako/Irodori-TTS-Server.git
cd Irodori-TTS-Server
uv sync --extra cu128
cp .env.example .env
```

Choose one PyTorch backend extra:

```bash
uv sync --extra cu128  # NVIDIA CUDA 12.8
uv sync --extra rocm   # AMD ROCm on Linux
uv sync --extra cpu    # CPU-only
```

The PyTorch backend extras are mutually exclusive. The `cu128` extra uses the PyTorch CUDA 12.8 index, the `rocm` extra uses the PyTorch ROCm index on Linux, and the `cpu` extra uses the CPU PyTorch index on Linux/Windows.

After syncing with a backend extra, use `uv run --no-sync ...` for the commands
below to avoid re-syncing the environment without the selected PyTorch backend
extra.

By default, the server downloads [`Aratako/Irodori-TTS-500M-v3`](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) from Hugging Face when the model is first loaded. To use a local checkpoint, set:

```bash
IRODORI_CHECKPOINT=/path/to/model.safetensors
```

## Running

```bash
uv run --no-sync python -m irodori_openai_tts --host 0.0.0.0 --port 8088
```

This uses the PyTorch backend selected during `uv sync`.

Open the health endpoint:

```bash
curl http://localhost:8088/health
```

## Docker

Create `.env` first:

```bash
cp .env.example .env
```

Set the backend used when the image is built:

```env
IRODORI_TTS_BACKEND=cu128
```

Supported values are `cu128`, `rocm`, and `cpu`.

On the first run, or after updating the server code, build and recreate the container:

```bash
docker compose up --build --force-recreate
```

After that, start the existing image normally:

```bash
docker compose up
```

For NVIDIA GPU settings, build and recreate with both Compose files:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build --force-recreate
```

Then use this for normal GPU startup:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up
```

For AMD ROCm, set `IRODORI_TTS_BACKEND=rocm` in `.env`, then build and recreate with the ROCm Compose file:

```bash
docker compose -f compose.yaml -f compose.rocm.yaml up --build --force-recreate
```

Then use this for normal ROCm startup:

```bash
docker compose -f compose.yaml -f compose.rocm.yaml up
```

For CPU-only Docker images, set `IRODORI_TTS_BACKEND=cpu` in `.env` before building.

Reference voices placed in `./voices` are available inside the container. Downloaded Hugging Face files are kept in a Docker volume so they are reused across container recreations.

## Quick Usage

Put a reference voice in `voices/`. Files can be added before or after the server starts; the directory is scanned when a request resolves a voice.

```text
voices/
  sample.wav
```

Then call the speech endpoint:

```bash
curl http://localhost:8088/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "irodori-tts",
    "input": "こんにちは。これはIrodori-TTSのAPIテストです。",
    "voice": "sample",
    "response_format": "wav"
  }' \
  --output speech.wav
```

Using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8088/v1",
    api_key="not-used",
)

with client.audio.speech.with_streaming_response.create(
    model="irodori-tts",
    voice="sample",
    input="こんにちは。これはIrodori-TTSのAPIテストです。",
    response_format="wav",
) as response:
    response.stream_to_file("speech.wav")
```

The SDK method name contains `streaming_response`, but this server still generates a complete response internally.

## API

### `GET /health`

Returns server status and current configuration. This endpoint does not load the model.

### `GET /v1/models`

Returns the model ID accepted by the speech endpoint.

Example response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "irodori-tts",
      "object": "model",
      "created": 0,
      "owned_by": "irodori-tts"
    }
  ]
}
```

### `POST /v1/audio/speech`

Synthesizes speech and returns audio bytes.

Request fields:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `model` | string | yes | Use `irodori-tts` unless you changed `IRODORI_MODEL_NAME`. |
| `input` | string | yes | Text to synthesize. |
| `voice` | string or object | no | Voice ID, or `{ "id": "voice_id" }`. Uses `IRODORI_DEFAULT_VOICE` if omitted. |
| `response_format` | string | no | `wav`, `mp3`, `flac`, `opus`, `aac`, or `pcm`. |
| `speed` | number | no | Speaking speed, from `0.25` to `4.0`. Higher is faster; internally this is converted to an inverse duration scale. |
| `stream_format` | string | no | Set to `sse` to receive chunk-level Server-Sent Events. |
| `irodori` | object | no | Irodori-specific inference options. |

When `stream_format: "sse"` is set, the response is `text/event-stream`.
The server synthesizes each text chunk sequentially and emits one `audio_chunk`
event per chunk, followed by a final `done` event:

For consistent voice tone across chunks, specify a reference voice with `voice`
or `irodori.ref_wav`. Without a reference, each chunk is synthesized
independently and the perceived voice tone may vary between chunks.

```bash
curl -N http://localhost:8088/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "irodori-tts",
    "input": "最初の文です。次の文です。",
    "voice": "sample",
    "response_format": "wav",
    "stream_format": "sse",
    "irodori": {
      "chunking_enabled": true,
      "chunk_min_chars": 1
    }
  }'
```

```text
event: audio_chunk
data: {"index":0,"text":"最初の文です。","format":"wav","media_type":"audio/wav","audio_base64":"...","seed":123,"total_to_decode":0.1}

event: audio_chunk
data: {"index":1,"text":"次の文です。","format":"wav","media_type":"audio/wav","audio_base64":"...","seed":123,"total_to_decode":0.1}

event: done
data: {"chunks":2}
```

Each `audio_base64` value contains a complete audio file for that chunk, so
clients can decode and enqueue chunks while later chunks are still generating.

Irodori-specific options:

```json
{
  "model": "irodori-tts",
  "input": "こんにちは。",
  "voice": "sample",
  "response_format": "wav",
  "speed": 1.1,
  "irodori": {
    "num_steps": 24,
    "cfg_scale_text": 3.0,
    "cfg_scale_speaker": 5.0,
    "lora_adapter": "/models/adapters/speaker-a",
    "seed": 1234,
    "t_schedule_mode": "sway",
    "sway_coeff": -1.0
  }
}
```

Common `irodori` options:

| Field | Notes |
| --- | --- |
| `num_steps` | Number of diffusion steps. Higher can improve quality but takes longer. |
| `seed` | Fixed random seed for reproducible output. |
| `cfg_scale_text` | Strength of text guidance. |
| `cfg_scale_speaker` | Strength of speaker/reference-voice guidance. |
| `lora_adapter` | PEFT LoRA adapter directory to load dynamically for this request. The adapter is not merged into the base checkpoint. |
| `t_schedule_mode` | Sampling schedule, usually `linear` or `sway`. |
| `sway_coeff` | Sway schedule coefficient when using `t_schedule_mode: "sway"`. |
| `lora_hot_swap` | Swap LoRA adapter weights in place so cached CUDA graphs survive the switch. Default from `IRODORI_DEFAULT_LORA_HOT_SWAP`. Refused automatically for incompatible adapters (DoRA, or `modules_to_save` beyond `duration_predictor`). |
| `apply_watermark` | Set `false` to skip the SilentCipher AI-generation watermark (~15 ms per request). Default from `IRODORI_DEFAULT_APPLY_WATERMARK`. |
| `chunking_enabled` | Enable or disable automatic long text chunking for this request. |
| `chunk_min_chars` | Minimum non-space characters before a chunk split point is used. |
| `first_sentence_chunk_min_chars` | Optional minimum non-space characters used only for splitting the first sentence. |
| `caption` | Voice/style description for caption-enabled VoiceDesign checkpoints. Ignored by checkpoints without caption conditioning. |
| `cfg_scale_caption` | Strength of caption guidance. |
| `max_caption_len` | Optional maximum caption token length. |

Dynamic LoRA loading is per runtime process. The first request for an adapter loads it into memory; later requests for the same adapter reuse the cached adapter. To run the base model after an adapter has been loaded, omit `lora_adapter` or set it to `null`, `"none"`, or `"base"`. Dynamic LoRA is not compatible with `IRODORI_COMPILE_MODEL=true`.

Switching `lora_adapter` normally drops all cached CUDA graphs (they are recaptured on the next requests). Set `lora_hot_swap: true` (or `IRODORI_DEFAULT_LORA_HOT_SWAP=true`) to swap adapter weights in place and keep the graphs; a tiny floating-point drift can accumulate per swap.

### `POST /v1/admin/prewarm`

Loads the model (if not loaded yet) and captures CUDA graphs so following requests take the fast replay path from the first request. Useful after startup, or to prewarm a specific LoRA adapter without restarting. Returns a summary string from the runtime.

All body fields are optional; omitted values fall back to the server defaults (`IRODORI_PREWARM_MAX_SECONDS`, `IRODORI_PREWARM_LORA_ADAPTER`, `IRODORI_DEFAULT_LORA_HOT_SWAP`, `IRODORI_DEFAULT_NUM_STEPS`).

```bash
curl -X POST http://localhost:8088/v1/admin/prewarm \
  -H "Content-Type: application/json" \
  -d '{
    "lora_adapter": "/models/adapters/speaker-a",
    "lora_hot_swap": true,
    "max_seconds": 15
  }'
```

The request occupies a synthesis slot while running, so concurrent speech requests wait for it (or time out after `IRODORI_SYNTHESIS_WAIT_TIMEOUT`). On non-CUDA devices, or with `IRODORI_DISABLE_CUDA_GRAPH=1`, prewarming is skipped and the status message says so.

### Voice Management

The server scans `IRODORI_VOICES_DIR` for voice files. File stems become voice IDs.

Supported audio extensions:

- `.wav`
- `.flac`
- `.mp3`
- `.m4a`
- `.ogg`
- `.opus`
- `.aac`
- `.webm`

Latent references and Speaker Inversion are also supported:

- `.pt`
- `.pth`
- `.speaker.safetensors`

Examples:

```text
voices/
  alice.wav      -> voice: "alice"
  bob.flac       -> voice: "bob"
  cached.pt      -> voice: "cached"
```

You can also create `voices/voices.json`:

```json
{
  "alice": "alice.wav",
  "bob": "bob_reference.flac",
  "cached": "cached.pt"
}
```

Text-only inference is available with `voice: "none"` when `IRODORI_ALLOW_NO_REF_VOICE=true`.

Voice file endpoints:

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/v1/audio/voices` | List resolved voices. |
| `POST` | `/v1/audio/voices` | Upload voice file with multipart `file` and optional `voice_id`. |
| `GET` | `/v1/audio/voices/{voice_id}` | Get uploaded voice file metadata. |
| `PUT` | `/v1/audio/voices/{voice_id}` | Replace uploaded voice file. |
| `DELETE` | `/v1/audio/voices/{voice_id}` | Delete uploaded voice file. |

Upload example:

```bash
curl http://localhost:8088/v1/audio/voices \
  -F voice_id=sample \
  -F file=@sample.wav
```

## Long Text Chunking

Long text chunking is enabled by default.

When enabled, the server splits text only when both conditions are met:

- the current chunk has at least `chunk_min_chars` non-space characters
- the current character is punctuation or a line break

Set `irodori.first_sentence_chunk_min_chars` to use a smaller threshold only
for the first sentence. Later sentences keep the normal `chunk_min_chars`
threshold.

Each chunk is synthesized sequentially, then concatenated into one audio response.

Per-request override:

```json
{
  "model": "irodori-tts",
  "input": "長い本文...",
  "voice": "sample",
  "response_format": "wav",
  "irodori": {
    "chunking_enabled": true,
    "chunk_min_chars": 80,
    "first_sentence_chunk_min_chars": 1
  }
}
```

If `irodori.seconds` is set, chunking is skipped because that fixed duration applies to the whole request.

## Request Queue

Only one synthesis request runs at a time by default. Additional requests wait for an available slot.

You can tune the queue with:

```env
IRODORI_MAX_CONCURRENT_SYNTHESIS=1
IRODORI_SYNTHESIS_WAIT_TIMEOUT=300
```

If the model is still loading or no synthesis slot becomes available before the configured timeout, the server returns HTTP 503.

## Performance

Inference runs on the [stak/Irodori-TTS performance fork](https://github.com/stak/Irodori-TTS). Most optimizations (TF32 matmul, CUDA graph replay, fp16 codec decode, text-length bucketing, request-time LoRA merge) are enabled by default on CUDA and fall back to upstream behavior on other devices. Full details, measurements, and opt-outs are in the fork's [docs/performance.md](https://github.com/stak/Irodori-TTS/blob/main/docs/performance.md).

Recommended server setup on an NVIDIA GPU:

1. Set `IRODORI_MODEL_PRECISION=bf16` and keep `IRODORI_CODEC_PRECISION=fp32` (the codec decoder already runs its fp16 fast path; bf16/fp16 codec precision only lowers quality). `compose.gpu.yaml` applies this pairing by default.
2. Set `IRODORI_PREWARM=true` so the server captures CUDA graphs during startup (this implies loading the model at startup). The first real request then takes the fast path instead of paying capture cost. A 15 s prewarm takes roughly 1 minute once and holds about 1-1.5 GiB of extra VRAM.
3. If requests use a LoRA adapter, set `IRODORI_PREWARM_LORA_ADAPTER` (or call `POST /v1/admin/prewarm` with `lora_adapter`) so the graphs are captured with that adapter loaded, and consider `IRODORI_DEFAULT_LORA_HOT_SWAP=true` so adapter switches keep the cached graphs.
4. On Linux/WSL2 (including the Docker image), `IRODORI_COMPILE=1` additionally runs the model through `torch.compile` inside the CUDA graphs for a further speedup. The first prewarm then takes a few minutes (later restarts are faster via the inductor on-disk cache). Leave it off on Windows-native.

Prewarmed graphs are keyed by tensor shapes and CFG scales: the server prewarms with its default sampling settings, so requests that override `cfg_scale_text`/`cfg_scale_speaker`, use `num_candidates > 1`, or supply reference audio capture their own graphs on first use (one-time, about a second each). `num_steps`, `seed`, `t_schedule_mode`, and `sway_coeff` can vary freely without recapture.

The performance-fork environment variables (`IRODORI_DISABLE_TF32`, `IRODORI_TEXT_BUCKETS`, ...) are read by the `irodori-tts` library directly from the process environment. The server exports `.env` to the environment at startup, so they can be configured in the same `.env` file; they are also listed in `.env.example`, `compose.gpu.yaml`, and `compose.rocm.yaml` with their defaults.

## Configuration

Server defaults are configured with environment variables. For local runs and Docker Compose, copy `.env.example` to `.env` and edit it as needed.

All environment variables use the `IRODORI_` prefix. Request fields override these defaults when the corresponding option is provided in the API request.

| Variable | Default | Notes |
| --- | --- | --- |
| `IRODORI_HOST` | `0.0.0.0` | Server host. |
| `IRODORI_PORT` | `8088` | Server port. |
| `IRODORI_TTS_BACKEND` | `cu128` | Docker build backend: `cu128`, `rocm`, or `cpu`. |
| `IRODORI_API_KEY` | unset | Optional bearer token. |
| `IRODORI_MODEL_NAME` | `irodori-tts` | Model ID used in requests. |
| `IRODORI_HF_CHECKPOINT` | `Aratako/Irodori-TTS-500M-v3` | Hugging Face repo containing `model.safetensors`. |
| `IRODORI_CHECKPOINT` | unset | Local checkpoint path. Takes precedence over `IRODORI_HF_CHECKPOINT`. |
| `IRODORI_CODEC_REPO` | `Aratako/Semantic-DACVAE-Japanese-32dim` | DACVAE codec repo or path. |
| `IRODORI_MODEL_DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. |
| `IRODORI_CODEC_DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. |
| `IRODORI_MODEL_PRECISION` | `fp32` | `fp32` or `bf16`. |
| `IRODORI_CODEC_PRECISION` | `fp32` | `fp32` or `bf16`. |
| `IRODORI_COMPILE_MODEL` | `false` | Enable `torch.compile` for core inference methods. Keep disabled when using dynamic LoRA adapters. |
| `IRODORI_COMPILE_DYNAMIC` | `false` | Use `dynamic=True` for `torch.compile`. |
| `IRODORI_PRELOAD` | `false` | Load the model during startup. |
| `IRODORI_PREWARM` | `false` | Capture CUDA graphs during startup so the first request is already fast. Implies loading the model at startup. CUDA only; a no-op elsewhere. |
| `IRODORI_PREWARM_MAX_SECONDS` | `15` | Upper bound of the audio duration range covered by prewarmed graphs. |
| `IRODORI_PREWARM_LORA_ADAPTER` | unset | LoRA adapter directory loaded before prewarming so its requests take the fast path immediately. |
| `IRODORI_MODEL_LOAD_TIMEOUT` | `300` | Seconds to wait for model loading. |
| `IRODORI_MAX_CONCURRENT_SYNTHESIS` | `1` | Maximum simultaneous synthesis jobs. |
| `IRODORI_SYNTHESIS_WAIT_TIMEOUT` | `300` | Seconds to wait for a synthesis slot. |
| `IRODORI_VOICES_DIR` | `voices` | Directory scanned for reference voices. |
| `IRODORI_DEFAULT_VOICE` | unset | Used when request omits `voice`. |
| `IRODORI_ALLOW_NO_REF_VOICE` | `true` | Allow `voice: "none"` text-only inference. |
| `IRODORI_DEFAULT_RESPONSE_FORMAT` | `wav` | Default response format. |
| `IRODORI_DEFAULT_NUM_STEPS` | `40` | Default diffusion steps. |
| `IRODORI_DEFAULT_T_SCHEDULE_MODE` | `linear` | Default timestep schedule. |
| `IRODORI_DEFAULT_SWAY_COEFF` | `-1.0` | Default sway coefficient. Used only when `t_schedule_mode` is `sway`. |
| `IRODORI_DEFAULT_DURATION_SCALE` | `1.0` | Default duration scale. |
| `IRODORI_DEFAULT_CFG_SCALE_TEXT` | `3.0` | Default text CFG scale. |
| `IRODORI_DEFAULT_CFG_SCALE_SPEAKER` | `5.0` | Default speaker CFG scale. |
| `IRODORI_DEFAULT_CFG_GUIDANCE_MODE` | `independent` | Default CFG guidance mode. |
| `IRODORI_DEFAULT_CHUNKING_ENABLED` | `true` | Enable punctuation-aware chunking by default. |
| `IRODORI_DEFAULT_CHUNK_MIN_CHARS` | `80` | Minimum non-space characters before a split point is used. |
| `IRODORI_DEFAULT_FIRST_SENTENCE_CHUNK_MIN_CHARS` | unset | Minimum non-space characters before the first sentence split point is used. Unset keeps normal `chunk_min_chars` behavior. |
| `IRODORI_DEFAULT_LORA_HOT_SWAP` | `false` | Swap LoRA adapter weights in place on adapter switches so cached CUDA graphs survive. |
| `IRODORI_DEFAULT_APPLY_WATERMARK` | `true` | Embed the SilentCipher AI-generation watermark in generated audio. |

The following variables are read by the [Irodori-TTS performance fork](https://github.com/stak/Irodori-TTS/blob/main/docs/performance.md) directly from the process environment (defaults shown; all optimizations are inference-only):

| Variable | Default | Notes |
| --- | --- | --- |
| `IRODORI_DISABLE_TF32` | `0` | `1` disables TF32 matmul (exact fp32, slower). |
| `IRODORI_DISABLE_LORA_MERGE` | `0` | `1` keeps LoRA adapters unmerged (upstream behavior). |
| `IRODORI_DISABLE_CUDA_GRAPH` | `0` | `1` disables CUDA graph capture/replay entirely. |
| `IRODORI_DISABLE_DURATION_GRAPH` | `0` | `1` keeps condition encoding + duration prediction eager (sampler graphs unaffected). |
| `IRODORI_DISABLE_FP16_DECODE` | `0` | `1` keeps the codec decoder in fp32 (exact decode, ~2x slower). |
| `IRODORI_TEXT_BUCKETS` | `64` | Comma-separated text-length buckets (tokens); short texts are padded to the smallest fitting bucket. `0` or empty disables. |
| `IRODORI_COMPILE` | `0` | `1` runs the model through `torch.compile` inside the CUDA step graphs. Requires a Triton toolchain (Linux/WSL2 + C compiler). |
| `IRODORI_CUDA_GRAPH_BUCKET` | `16` | Latent-length bucket size in patched steps; `1` disables padding. |
| `IRODORI_CUDA_GRAPH_CACHE` | `64` | Maximum cached CUDA graph entries. |

## Development

Run tests:

```bash
uv run --extra dev pytest
```

Run lint:

```bash
uv run --extra dev ruff check src tests
```

Run import/bytecode checks:

```bash
uv run python -m compileall src tests
```

## License

This server code is released under the MIT License. See [LICENSE](LICENSE).

Model weights and codec assets are distributed separately. Check the Hugging Face model cards for their licenses and usage terms:

- [Aratako/Irodori-TTS-500M-v3](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)
- [Aratako/Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim)
