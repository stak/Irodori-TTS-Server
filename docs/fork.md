# Fork Notes

This repository is a fork of [Aratako/Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server).

Everything that behaves differently here — extra request fields, extra environment
variables, extra response data, and the Docker/Compose differences — is collected in this
file so that [README.md](../README.md) stays close to upstream and upstream changes stay
easy to merge.

## Contents

- [Dependency: the inference fork](#dependency-the-inference-fork)
- [Performance profile](#performance-profile)
- [Best-of-N candidates](#best-of-n-candidates)
- [Speaker Inversion blending](#speaker-inversion-blending)
- [`duration_ignore_speaker`](#duration_ignore_speaker)
- [LoRA hot-swap](#lora-hot-swap)
- [Watermark toggle](#watermark-toggle)
- [MP3 quality settings](#mp3-quality-settings)
- [Chunking differences and explicit chunks](#chunking-differences-and-explicit-chunks)
- [Response headers](#response-headers)
- [SSE `audio_chunk` payload](#sse-audio_chunk-payload)
- [Browser test client](#browser-test-client)
- [Serving a client from this server](#serving-a-client-from-this-server)
- [Docker and Compose differences](#docker-and-compose-differences)
- [Fork environment variables](#fork-environment-variables)
- [Appendix A: request fields the README does not list](#appendix-a-request-fields-the-readme-does-not-list)
- [Appendix B: environment variables the README does not list](#appendix-b-environment-variables-the-readme-does-not-list)
- [Appendix C: behavior worth knowing](#appendix-c-behavior-worth-knowing)

## Dependency: the inference fork

The server does not run on upstream Irodori-TTS. `pyproject.toml` declares:

```toml
"irodori-tts @ git+https://github.com/stak/Irodori-TTS.git",
```

There is no `rev=` in `pyproject.toml`; the exact commit is pinned **only in `uv.lock`**
(`source = { git = "https://github.com/stak/Irodori-TTS.git#<commit>" }`). Two consequences:

- Reproducibility comes from the lock file, not the requirement. `uv sync` honours the
  lock, but `uv sync --upgrade` (or a deleted/regenerated lock) moves the dependency to
  whatever `main` currently is. `Dockerfile` builds with `uv sync --locked`, so image
  builds always match the lock.
- This is one reason the README tells you to run commands with `uv run --no-sync`: a bare
  `uv run` may re-sync the environment, which also drops the PyTorch backend extra you
  selected.

The inference fork adds CUDA-graph replay, `torch.compile` inside the graphs, fp16 codec
decode, text-length bucketing, request-time LoRA merge, LoRA hot-swap, and an optional
watermark toggle. Its own documentation lives in
[stak/Irodori-TTS docs/performance.md](https://github.com/stak/Irodori-TTS/blob/main/docs/performance.md).

## Performance profile

The inference fork groups its optimizations behind `IRODORI_PERF_PROFILE`. The library
default is `upstream` (bit-identical to unmodified Irodori-TTS); **this server exports
`recommended` by default**, from its own `IRODORI_RUNTIME_PROFILE` setting, because
serving fast is its purpose.

- `IRODORI_RUNTIME_PROFILE=upstream` serves upstream-identical outputs.
- An `IRODORI_PERF_PROFILE` that is already present in the environment always wins over
  `IRODORI_RUNTIME_PROFILE`.
- Individual `IRODORI_DISABLE_*` variables override either profile.

Recommended setup on an NVIDIA GPU:

1. Set `IRODORI_MODEL_PRECISION=bf16` and keep `IRODORI_CODEC_PRECISION=fp32`. The codec
   decoder already runs its own fp16 fast path, so bf16/fp16 codec precision only lowers
   quality. `compose.gpu.yaml` applies this pairing by default.
2. If requests use a LoRA adapter, consider `IRODORI_DEFAULT_LORA_HOT_SWAP=true` so
   adapter switches keep the cached graphs.
3. On Linux/WSL2 (including the Docker image), `IRODORI_COMPILE=1` additionally runs the
   model through `torch.compile` inside the CUDA graphs. Run the inference fork's
   `precompile.py` once beforehand with the production shape grid so no real request pays
   an on-demand compile; later restarts reuse the on-disk compile caches. In Docker those
   caches live in the `inductor_cache` / `triton_cache` volumes, so only the first
   container run pays the cold-compile cost. Leave it off on Windows-native.

CUDA graphs are captured lazily and keyed by tensor shapes and CFG scales: the first
request at a new shape (length bucket, candidate count, CFG scales, reference
conditioning) pays a one-time capture cost of about a second, and every following request
with that shape replays the graph. `num_steps`, `seed`, `t_schedule_mode`, and
`sway_coeff` can vary freely without recapture.

The performance-fork variables are read by the `irodori-tts` library directly from the
process environment, not through this server's settings. The server loads `.env` into the
environment at startup, so they can be configured in the same `.env` file; they are also
listed in `.env.example`, `compose.gpu.yaml`, and `compose.rocm.yaml` with their defaults.

## Best-of-N candidates

Set the OpenAI-style top-level `n` to generate several takes of the same input in one
batched sampling pass and receive all of them, e.g. to pick the best take downstream with
a speaker-similarity or quality scorer. Text encoding and reference-voice conditioning are
computed once and shared across the batch, so this is cheaper than sending `n` separate
requests. Candidate ranking is out of scope for the server; it returns every candidate and
leaves selection to the client.

```bash
curl http://localhost:8088/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "irodori-tts",
    "input": "こんにちは、今日はいい天気ですね。",
    "voice": "sample",
    "response_format": "wav",
    "n": 3
  }'
```

With `n > 1` the response is JSON instead of raw audio bytes:

```json
{
  "object": "speech.candidates",
  "seed": 1234,
  "sample_rate": 44100,
  "candidates": [
    {"index": 0, "audio": "<base64>", "format": "wav", "media_type": "audio/wav", "seed": 1234, "duration_sec": 3.2},
    {"index": 1, "audio": "<base64>", "format": "wav", "media_type": "audio/wav", "seed": 1235, "duration_sec": 3.4},
    {"index": 2, "audio": "<base64>", "format": "wav", "media_type": "audio/wav", "seed": 1236, "duration_sec": 3.1}
  ]
}
```

Note that the base64 audio field is named `audio` here, while the SSE path uses
`audio_base64`.

Seed semantics: candidate `i` draws its initial noise from its own generator seeded with
`base seed + i`. The response's top-level `seed` is the base seed and each candidate
carries its own derived `seed`. The `X-Irodori-Seed` response header also carries the base
seed, not a per-candidate list. Two ways to bring a chosen candidate back:

- **Standalone regeneration** — re-send the request with `n: 1` and `irodori.seed` set to
  the candidate's `seed`. This draws bit-identical initial noise, i.e. the same take. It is
  NOT bit-exact against the batched output on CUDA: kernel numerics differ between batch
  sizes and the diffusion steps amplify them chaotically. Measured on an RTX 4090 (bf16):
  duration always identical, waveform correlation 0.88–0.99 depending on the draw — treat
  it as "the same take, re-rendered", not as a copy. When the archived bytes must match,
  use the bit-exact path below.
- **Bit-exact reproduction** — re-send the identical request (same `irodori.seed`, same
  `n`) and take the same `index`. Warm responses reproduce bit-exactly; only the very first
  request at a new batch shape (CUDA graph capture) can deviate negligibly (measured max
  abs sample difference 1.6e-3, correlation 1.000000).

Per-candidate seeds require an `irodori-tts` build with the per-candidate-seeds patch
(candidate noise drawn per row from `seed + i`); with older builds all candidates share
the base seed and only the bit-exact path applies.

Restrictions with `n > 1`:

- Automatic text chunking is skipped; the whole `input` is synthesized in a single pass
  (so it is subject to `max_seconds`). Intended for short lines.
- Explicit `irodori.chunks` and `stream_format: "sse"` are rejected with `400`.
- `n` is capped by `IRODORI_MAX_NUM_CANDIDATES` (default `8`); `n < 1` or `n` above the cap
  is a `400`. All candidates are sampled in one batch, so peak VRAM and per-step compute
  grow with `n`; see the measured numbers below before raising the cap.

`irodori.num_candidates` is accepted as an alias (it maps to the same runtime field). This
is the one option where the top-level key wins: `n` takes precedence over
`irodori.num_candidates` when both are set. Without either, `IRODORI_DEFAULT_NUM_CANDIDATES`
(default `1`) applies.

Measured on an RTX 4090 (bf16 model, fp32 codec, CUDA graphs enabled, 40 steps, ~6.5 s of
48 kHz output per candidate, warm graphs):

| `n` | Latency (warm) | `n` separate warm requests | Process VRAM |
| --- | --- | --- | --- |
| 1 | 0.53 s | – | 8.7 GiB |
| 2 | 0.77 s | 1.06 s | 8.7 GiB |
| 4 | 1.21 s | 2.12 s | 8.8 GiB |
| 8 | 2.32 s | 4.24 s | 8.8 GiB |

As with any other change in tensor shapes, the first request at a new batch size captures
CUDA graphs once (~3–4 s extra); later requests with the same `n` replay them.

## Speaker Inversion blending

Two or more Speaker Inversion voices can be blended at request time with
`irodori.ref_embeds`, so mix ratios can be auditioned freely without baking intermediate
files:

```json
{
  "model": "irodori-tts",
  "input": "こんにちは",
  "irodori": {
    "ref_embeds": [
      {"voice": "alice", "weight": 0.7},
      {"voice": "bob", "weight": 0.3}
    ]
  }
}
```

- Components reference registered voice IDs that resolve to `.speaker.safetensors` files.
  Raw paths are not accepted, and an unknown key inside a component is rejected.
- Weights must be `>= 0` and are normalized to sum to 1. `weight: 0` components are
  dropped, so a ratio slider can keep sending every component at either extreme; a list
  where every weight is `0` is a `400`.
- `ref_embeds` cannot be combined with `ref_wav` / `ref_latent` / `ref_embed` / `no_ref`
  (`400`). When it is used, the request's `voice` is ignored.
- The optional `irodori.si_blend_mode` selects the blend algorithm: `lerp` (default,
  weighted mean; requires equal token counts and preserves conditioning shapes, so cached
  CUDA graphs are reused across ratio changes) or `concat` (experimental token
  concatenation, which allows differing token counts).
- The blend is deterministic: given the same weights it is bit-identical to pre-baking the
  file with Irodori-TTS's `blend_speaker_embeddings.py` and passing it as a voice.

## `duration_ignore_speaker`

`irodori.duration_ignore_speaker` predicts length from the unconditional speaker token, so
the speaker condition (Speaker Inversion embedding or reference audio) drives timbre only,
not duration. Default from `IRODORI_DEFAULT_DURATION_IGNORE_SPEAKER` (`false`).

This is useful when the duration predictor was trained speaker-unconditioned — for example
a LoRA baked without speaker conditioning so that Speaker Inversion stays applicable.
Supplying any speaker condition then routes duration through a path the predictor never
trained on and shifts predicted length off the intended pacing, which this option restores.

## LoRA hot-swap

Switching `irodori.lora_adapter` normally drops all cached CUDA graphs (they are
recaptured on the next requests). Set `irodori.lora_hot_swap: true` (or
`IRODORI_DEFAULT_LORA_HOT_SWAP=true`) to swap adapter weights in place and keep the
graphs. A tiny floating-point drift can accumulate per swap. Hot-swap is refused
automatically for incompatible adapters (DoRA, or `modules_to_save` beyond
`duration_predictor`), which falls back to the normal reload path.

## Watermark toggle

`irodori.apply_watermark: false` skips the SilentCipher AI-generation watermark (~15 ms
per request). Default from `IRODORI_DEFAULT_APPLY_WATERMARK` (`true`).

## MP3 quality settings

MP3 encoding quality is configurable, and the default is higher-quality than libsndfile's
own default:

| Variable | Default | Notes |
| --- | --- | --- |
| `IRODORI_MP3_BITRATE_MODE` | `VARIABLE` | `VARIABLE` (LAME VBR), `AVERAGE`, or `CONSTANT`. |
| `IRODORI_MP3_COMPRESSION_LEVEL` | `0.0` | `0.0` (best) to just under `1.0`. For `VARIABLE` this maps to the LAME `-V` quality (`0.0` = `-V0`, ~160 kbps mono speech); for `CONSTANT` it selects the bitrate (`0.0` = 320 kbps). |

Both apply to the default soundfile encoder. The torchaudio/ffmpeg fallbacks use their own
defaults.

## Chunking differences and explicit chunks

Upstream splits on any punctuation. This fork splits only on sentence-ending punctuation
(`。．.!！?？`) or a line break. Commas (`、，,`) are intentionally not split points:
Irodori-TTS conditions its output on the whole text of a chunk, so splitting mid-sentence
loses cross-chunk nuance.

`irodori.chunk_pause_seconds` inserts that much silence at each join when the server
concatenates chunks into one response (default `0`, i.e. back-to-back, the upstream
behavior). Tail silence is already trimmed by `trim_tail`, so the inserted pause is
well-defined. It is not applied on the SSE path, where the client receives chunks
individually.

### Explicit chunks

Because prosody depends on where the split lands, the boundaries can be passed explicitly
with `irodori.chunks` when the automatic splitter picks bad ones:

```json
{
  "model": "irodori-tts",
  "input": "一文目。二文目はとても長い…。三文目。",
  "voice": "none",
  "irodori": {
    "chunks": ["一文目。二文目はとても長い…。", "三文目。"],
    "chunking_enabled": false
  }
}
```

- Each entry is a hard boundary: the server never merges entries, and text never crosses
  an entry boundary.
- `input` remains required (OpenAI SDK compatibility) but is not synthesized when `chunks`
  is present — set it to the joined text or a placeholder.
- With `chunking_enabled: true` (the default) each entry may still be split further by the
  automatic splitter; send `chunking_enabled: false` for exact control.
- Cannot be combined with `irodori.seconds` (`400`): a single fixed duration is ambiguous
  across chunks.
- The entries' combined length is limited to 4096 characters (`400` above that), the same
  limit `input` has. The two limits are independent, so a request may legally carry up to
  ~8192 characters across `input` plus `chunks`.

## Response headers

Successful audio responses carry timing and seed metadata in headers. This applies to the
normal audio response and to the `n > 1` JSON candidate response; SSE responses carry none
of them (they report the same values inside each event instead).

| Header | Notes |
| --- | --- |
| `X-Irodori-Seed` | The seed actually used, as an integer. With `n > 1` this is the base seed, not a per-candidate list. |
| `X-Irodori-Total-To-Decode` | Seconds spent up to the decode step, formatted with 6 decimals. |
| `X-Irodori-Encode-Seconds` | Seconds spent encoding the output audio format, formatted with 6 decimals. Fork addition. |
| `X-Irodori-Messages` | Runtime notices joined with `" \| "`, truncated to 4096 characters. Omitted when there are no messages. |

Browser clients on another origin can read these: the fork adds all four to the CORS
`expose_headers` list. That list only takes effect when the CORS middleware is installed,
i.e. when `IRODORI_CORS_ORIGINS` is non-empty.

## SSE `audio_chunk` payload

Each `audio_chunk` event carries the following fields, one more than the README's example
shows:

| Field | Notes |
| --- | --- |
| `index` | 0-based chunk index. |
| `text` | The chunk text that was synthesized. |
| `format` | Response format of this chunk. |
| `media_type` | Content type matching `format`. |
| `audio_base64` | Complete audio file for the chunk, base64-encoded. |
| `seed` | Seed used for the chunk. |
| `total_to_decode` | Seconds spent up to the decode step. |
| `encode_seconds` | Seconds spent encoding this chunk, rounded to 6 decimals. Fork addition. |

The stream ends with `event: done`, `data: {"chunks": <count>}`. Errors raised mid-stream
are emitted as `event: error` with an OpenAI-style error object.

## Browser test client

[examples/test-client.html](../examples/test-client.html) is a self-contained browser page
(a fork addition) for exercising the API by hand. Generated audio plays inline and can be
downloaded, and a request history pane keeps the response details.

Controls it exposes:

- server URL and API key
- health check and voice list refresh
- input text, voice selection (including `none`)
- `num_steps`, `t_schedule_mode` (`linear` / `sway`), `sway_coeff`, `seed`, `speed`
- `response_format` (defaults to `mp3` here)
- Speaker Inversion blending: an on/off toggle, `si_blend_mode`, and any number of
  voice+weight rows (sent as `irodori.ref_embeds`; the plain `voice` field is dropped while
  blending is active)
- `lora_adapter` path and `lora_hot_swap`
- `apply_watermark`
- SSE streaming with a per-chunk arrival log
- chunking: `chunking_enabled`, `chunk_min_chars`, `chunk_pause_seconds`, and a mode that
  splits the textarea on blank lines into explicit `irodori.chunks`

It does **not** expose the two newest fork features: there is no Best-of-N (`n` /
`num_candidates`) control and no `duration_ignore_speaker` control, so those have to be
exercised with `curl` or an SDK.

Open the file directly in a browser. Because it then calls the API from a `file://`
origin, allow it in the server configuration:

```env
IRODORI_CORS_ORIGINS=["*"]
```

Then point the "サーバー URL" field at your server (default `http://127.0.0.1:8088`) and
press ヘルスチェック.

## Serving a client from this server

A `file://` page has a null origin, so every request carrying `Authorization` or
`Content-Type: application/json` must be preflighted. Proxies and endpoint security
products sometimes drop those `OPTIONS` requests without replying, which looks like a
working health check (a header-less `GET` is a simple request and needs no preflight)
followed by synthesis that hangs or fails. Correct CORS configuration cannot help: the
preflight never reaches the server.

Serving the page from this server instead makes its API calls same-origin, which are never
preflighted at all. Point `IRODORI_STATIC_FILE` at any HTML file and choose the route to
serve it at:

```env
IRODORI_STATIC_FILE=/path/to/your-client.html
IRODORI_STATIC_ROUTE=/client
```

Details:

- The feature is off unless `IRODORI_STATIC_FILE` is set.
- `IRODORI_STATIC_ROUTE` defaults to `/`, must start with `/`, and must not collide with a
  built-in route; startup fails if it does or if the route is malformed.
- Only `GET` is registered for the route — no `HEAD` — and the route is hidden from the
  OpenAPI schema.
- The file is read per request, so editing it does not need a restart. If the path does not
  exist at request time the route answers `404` (the route is registered regardless).
- Responses are sent with `Cache-Control: no-cache`.
- A client served this way needs no `IRODORI_CORS_ORIGINS` entry, and `GET /health` reports
  the active setting under `static`.

To restrict who can load the page, guard the route with HTTP Basic auth:

```env
IRODORI_STATIC_AUTH_USER=operator
IRODORI_STATIC_AUTH_PASSWORD=choose-something-long
```

- Both must be set together; setting only one fails at startup rather than leaving the
  route unprotected.
- A missing or wrong credential gets `401` with `WWW-Authenticate: Basic realm="Restricted"`.
- This guards the hosted file only. The API still uses `IRODORI_API_KEY`, so requests from
  the page keep sending their bearer token and are unaffected; conversely the API key is
  not accepted on the static route and Basic credentials are not accepted on `/v1/...`.
- Guarding the page matters because a browser client normally carries the API key in its
  own source, so anyone who can load the file can also call the API.
- Basic auth sends credentials base64-encoded, not encrypted. Over plain HTTP they are
  readable by anyone on the path, so treat this as access control against casual reach, not
  as transport security; put the server behind TLS if that matters.

Note that `/health` is unauthenticated either way — see
[Appendix C](#appendix-c-behavior-worth-knowing).

### Serving the bundled client under Docker

The image copies only `README.md`, `LICENSE`, and `src`, and Compose mounts only `./voices`
plus the caches, so `examples/` does not exist inside the container and
`IRODORI_STATIC_FILE=examples/test-client.html` would answer `404`. Mount the directory to
serve it:

```yaml
# compose.override.yaml
services:
  api:
    volumes:
      - ./examples:/app/examples:ro
```

```env
IRODORI_STATIC_FILE=/app/examples/test-client.html
IRODORI_STATIC_ROUTE=/client
```

Any other host path works the same way, as long as the path in `IRODORI_STATIC_FILE` is
the path *inside* the container.

## Docker and Compose differences

`compose.gpu.yaml` and `compose.rocm.yaml` set defaults that differ from the plain
`compose.yaml` and from the documented server defaults. Every value is written as
`${VAR:-default}`, so anything already set in `.env` still wins.

| Variable | `compose.yaml` | `compose.gpu.yaml` | `compose.rocm.yaml` | Server default |
| --- | --- | --- | --- | --- |
| `IRODORI_MODEL_DEVICE` | – | `cuda` | `cuda` | `auto` |
| `IRODORI_CODEC_DEVICE` | – | `cuda` | `cuda` | `auto` |
| `IRODORI_MODEL_PRECISION` | – | `bf16` | `bf16` | `fp32` |
| `IRODORI_CODEC_PRECISION` | – | `fp32` | `bf16` | `fp32` |
| `IRODORI_PRELOAD` | – | `true` | `false` | `false` |
| `IRODORI_RUNTIME_PROFILE` | – | `recommended` | `recommended` | `recommended` |
| `IRODORI_COMPILE` | – | `0` | `0` | `0` (library) |
| `IRODORI_CUDA_GRAPH_BUCKET` | – | `16` | `16` | `16` (library) |
| `IRODORI_CUDA_GRAPH_CACHE` | – | `64` | `64` | `64` (library) |
| `IRODORI_DISABLE_*`, `IRODORI_TEXT_BUCKETS` | – | passed through, empty | passed through, empty | profile decides |

Two of these are worth calling out:

- **`IRODORI_PRELOAD=true` in `compose.gpu.yaml`** contradicts the server default of
  `false`: with the GPU Compose file the model loads during startup, so the container takes
  longer to become ready and holds VRAM even when idle. Set `IRODORI_PRELOAD=false` in
  `.env` to get the documented lazy behavior. `compose.rocm.yaml` keeps `false`.
- **`IRODORI_CODEC_PRECISION=bf16` in `compose.rocm.yaml`** is the exception to the
  "keep the codec in fp32" recommendation: the ROCm path uses bf16 there. On CUDA keep
  `fp32`, where the decoder's own fp16 fast path already provides the speedup.

`compose.yaml` itself sets no `environment:` block at all; it only loads `.env` via
`env_file`.

## Fork environment variables

Variables added by this fork, on top of the table in the README:

| Variable | Default | Notes |
| --- | --- | --- |
| `IRODORI_RUNTIME_PROFILE` | `recommended` | Exported to the `irodori-tts` library as `IRODORI_PERF_PROFILE` at startup (an already-set `IRODORI_PERF_PROFILE` wins). `upstream` serves bit-identical unmodified-Irodori-TTS outputs. |
| `IRODORI_MP3_BITRATE_MODE` | `VARIABLE` | See [MP3 quality settings](#mp3-quality-settings). |
| `IRODORI_MP3_COMPRESSION_LEVEL` | `0.0` | See [MP3 quality settings](#mp3-quality-settings). |
| `IRODORI_DEFAULT_DURATION_IGNORE_SPEAKER` | `false` | Default for `irodori.duration_ignore_speaker`. |
| `IRODORI_DEFAULT_NUM_CANDIDATES` | `1` | Default candidate count when the request omits `n` / `irodori.num_candidates`. |
| `IRODORI_MAX_NUM_CANDIDATES` | `8` | Upper bound accepted for `n` / `irodori.num_candidates`. |
| `IRODORI_DEFAULT_LORA_HOT_SWAP` | `false` | Default for `irodori.lora_hot_swap`. |
| `IRODORI_DEFAULT_APPLY_WATERMARK` | `true` | Default for `irodori.apply_watermark`. |
| `IRODORI_STATIC_FILE` | unset | Path to a single file (typically an HTML client) to serve. Static hosting is off while unset. |
| `IRODORI_STATIC_ROUTE` | `/` | Route that `IRODORI_STATIC_FILE` is served at. |
| `IRODORI_STATIC_AUTH_USER` | unset | HTTP Basic auth username for the static route. Must be set together with the password. |
| `IRODORI_STATIC_AUTH_PASSWORD` | unset | HTTP Basic auth password for the static route. |

These are read by the [inference fork](https://github.com/stak/Irodori-TTS/blob/main/docs/performance.md)
directly from the process environment, not through this server's settings (defaults shown;
all optimizations are inference-only):

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

## Appendix A: request fields the README does not list

Fork-added `irodori` options:

| Field | Default | Notes |
| --- | --- | --- |
| `num_candidates` | `IRODORI_DEFAULT_NUM_CANDIDATES` (`1`) | Alias of top-level `n`; `n` wins. See [Best-of-N candidates](#best-of-n-candidates). |
| `duration_ignore_speaker` | `IRODORI_DEFAULT_DURATION_IGNORE_SPEAKER` (`false`) | See [`duration_ignore_speaker`](#duration_ignore_speaker). |
| `lora_hot_swap` | `IRODORI_DEFAULT_LORA_HOT_SWAP` (`false`) | See [LoRA hot-swap](#lora-hot-swap). |
| `apply_watermark` | `IRODORI_DEFAULT_APPLY_WATERMARK` (`true`) | See [Watermark toggle](#watermark-toggle). |
| `ref_embeds` | none | Speaker Inversion blend components. See [Speaker Inversion blending](#speaker-inversion-blending). |
| `si_blend_mode` | `lerp` | `lerp` or `concat`. |
| `chunks` | none | Explicit chunk list. See [Explicit chunks](#explicit-chunks). |
| `chunk_pause_seconds` | `0` | Silence inserted at each chunk join (non-SSE only). |

The following `irodori` options exist upstream as well but are not listed in the README.
They are accepted and forwarded to the runtime. Defaults come from settings where a
setting exists; otherwise the runtime's own default applies.

| Field | Default | Notes |
| --- | --- | --- |
| `ref_latent` | none | Path to a precomputed reference latent. |
| `ref_embed` | none | Path to a Speaker Inversion embedding. |
| `no_ref` | `false` | Synthesize without any speaker conditioning. |
| `duration_scale` | `IRODORI_DEFAULT_DURATION_SCALE` (`1.0`) | Multiplier on predicted duration. Divided by `speed` when `speed != 1.0`. |
| `min_seconds` | `IRODORI_DEFAULT_MIN_SECONDS` (`0.5`) | Lower clamp on generated duration. |
| `max_seconds` | `IRODORI_DEFAULT_MAX_SECONDS` (`30.0`) | Upper clamp on generated duration. This is the `max_seconds` the chunking section refers to. |
| `max_ref_seconds` | `IRODORI_DEFAULT_MAX_REF_SECONDS` (`30.0`) | Reference audio is truncated to this length. |
| `ref_normalize_db` | `IRODORI_DEFAULT_REF_NORMALIZE_DB` (`-16.0`) | Loudness target applied to reference audio. |
| `ref_ensure_max` | `IRODORI_DEFAULT_REF_ENSURE_MAX` (`true`) | Peak-limit the reference after normalization. |
| `decode_mode` | `IRODORI_DEFAULT_DECODE_MODE` (`sequential`) | `sequential` or `batch` codec decoding. |
| `cfg_guidance_mode` | `IRODORI_DEFAULT_CFG_GUIDANCE_MODE` (`independent`) | `independent`, `joint`, or `alternating`. |
| `cfg_scale` | runtime default | Combined CFG scale, for guidance modes that use a single scale. |
| `cfg_min_t` | `IRODORI_DEFAULT_CFG_MIN_T` (`0.5`) | Lower timestep bound where CFG is applied. |
| `cfg_max_t` | `IRODORI_DEFAULT_CFG_MAX_T` (`1.0`) | Upper timestep bound where CFG is applied. |
| `truncation_factor` | runtime default | Truncates the sampled noise distribution. |
| `rescale_k` | runtime default | CFG rescaling strength. |
| `rescale_sigma` | runtime default | CFG rescaling sigma. |
| `context_kv_cache` | `IRODORI_DEFAULT_CONTEXT_KV_CACHE` (`true`) | Reuse the context KV cache across steps. |
| `speaker_kv_scale` | runtime default | Scales speaker KV contributions. |
| `speaker_kv_min_t` | runtime default | Lower timestep bound for speaker KV scaling. |
| `speaker_kv_max_layers` | runtime default | Limits speaker KV scaling to the first N layers. |
| `trim_tail` | `IRODORI_DEFAULT_TRIM_TAIL` (`true`) | Trim trailing silence. |
| `tail_window_size` | `IRODORI_DEFAULT_TAIL_WINDOW_SIZE` (`20`) | Window used by tail trimming. |
| `tail_std_threshold` | `IRODORI_DEFAULT_TAIL_STD_THRESHOLD` (`0.05`) | Std threshold used by tail trimming. |
| `tail_mean_threshold` | `IRODORI_DEFAULT_TAIL_MEAN_THRESHOLD` (`0.1`) | Mean threshold used by tail trimming. |
| `max_text_len` | runtime default | Optional cap on text token length. |
| `seconds` | none | Fixed output duration. Disables chunking, and is divided by `speed` when `speed != 1.0`. |

### Top-level extras

Every `irodori.*` option is also accepted as a **top-level** request key, because both the
request model and the options model allow extra keys. `{"duration_ignore_speaker": true}`
at the top level is equivalent to putting it inside `irodori`. This exists so that clients
which cannot nest custom fields (some SDK wrappers) can still reach the options, and the
test suite covers it.

Precedence: the value inside `irodori` wins over the top-level key for every option except
`num_candidates`, where the OpenAI-style top-level `n` deliberately wins. For most options
the comparison is "first non-null value", so an explicit `irodori.x: null` does *not*
suppress a top-level `x`; three options (`ref_normalize_db`, `max_ref_seconds`,
`first_sentence_chunk_min_chars`) are resolved by key presence instead, so for those an
explicit `null` is honoured. `chunking` is accepted as a top-level-only alias of
`chunking_enabled`.

Caveat: because both models allow extra keys, **an unknown or misspelled option is silently
ignored** rather than rejected. `{"irodori": {"num_stpes": 8}}` is a valid request that
simply uses the default step count. Check `X-Irodori-*` headers or the server log if an
option seems to have no effect.

## Appendix B: environment variables the README does not list

These settings exist upstream as well but are not in the README's table. They are listed
here rather than added to the README so that the README stays close to upstream.

| Variable | Default | Notes |
| --- | --- | --- |
| `IRODORI_CORS_ORIGINS` | unset | JSON list of allowed CORS origins, e.g. `["*"]`. The CORS middleware is only installed when this is non-empty; that is also what enables the `X-Irodori-*` header exposure. |
| `IRODORI_VOICE_ALIASES_FILE` | unset | Path to the voice alias JSON. Unset falls back to `<IRODORI_VOICES_DIR>/voices.json` when that file exists. |
| `IRODORI_CODEC_DETERMINISTIC_ENCODE` | `true` | Deterministic codec encode path. |
| `IRODORI_CODEC_DETERMINISTIC_DECODE` | `true` | Deterministic codec decode path. |
| `IRODORI_DEFAULT_MIN_SECONDS` | `0.5` | Default for `irodori.min_seconds`. |
| `IRODORI_DEFAULT_MAX_SECONDS` | `30.0` | Default for `irodori.max_seconds`. |
| `IRODORI_DEFAULT_CFG_MIN_T` | `0.5` | Default for `irodori.cfg_min_t`. |
| `IRODORI_DEFAULT_CFG_MAX_T` | `1.0` | Default for `irodori.cfg_max_t`. |
| `IRODORI_DEFAULT_CONTEXT_KV_CACHE` | `true` | Default for `irodori.context_kv_cache`. |
| `IRODORI_DEFAULT_MAX_REF_SECONDS` | `30.0` | Default for `irodori.max_ref_seconds`. |
| `IRODORI_DEFAULT_REF_NORMALIZE_DB` | `-16.0` | Default for `irodori.ref_normalize_db`. |
| `IRODORI_DEFAULT_REF_ENSURE_MAX` | `true` | Default for `irodori.ref_ensure_max`. |
| `IRODORI_DEFAULT_TRIM_TAIL` | `true` | Default for `irodori.trim_tail`. |
| `IRODORI_DEFAULT_TAIL_WINDOW_SIZE` | `20` | Default for `irodori.tail_window_size`. |
| `IRODORI_DEFAULT_TAIL_STD_THRESHOLD` | `0.05` | Default for `irodori.tail_std_threshold`. |
| `IRODORI_DEFAULT_TAIL_MEAN_THRESHOLD` | `0.1` | Default for `irodori.tail_mean_threshold`. |
| `IRODORI_DEFAULT_DECODE_MODE` | `sequential` | Default for `irodori.decode_mode`. |

There is no `IRODORI_DEFAULT_CFG_SCALE_CAPTION`; see
[Appendix C](#appendix-c-behavior-worth-knowing).

## Appendix C: behavior worth knowing

Not fork-specific, but undocumented and easy to be surprised by.

- **`/health` is unauthenticated.** Every `/v1/...` route requires the bearer token when
  `IRODORI_API_KEY` is set, but `/health` does not. It discloses the configured checkpoint
  repo, the resolved local checkpoint path once the model has loaded, the configured device
  and precision strings, the voices directory and its file list, the request defaults, and
  the static-hosting path and whether Basic auth is on (never the credentials themselves).
  Treat it as public information or block it at the reverse proxy.
- **`input` is limited to 4096 characters.** Longer input is rejected with `422` by request
  validation, not chunked. Empty input is also `422`, and whitespace-only input is `400`.
  The explicit `chunks` list has its own, independent 4096-character total limit (`400`).
- **`speed` also divides an explicit `irodori.seconds`.** `speed` is normally converted to
  an inverse duration scale, but when `irodori.seconds` is set as well, that fixed duration
  is divided by `speed` too — so `seconds: 4` with `speed: 2` targets 2 seconds. Send
  `speed: 1.0` when `seconds` should be taken literally.
- **`cfg_scale_caption` has no default of its own.** When the request omits it, it falls
  back to `IRODORI_DEFAULT_CFG_SCALE_TEXT` (`3.0`), so changing the text CFG default
  silently changes the caption CFG default as well. Set `irodori.cfg_scale_caption`
  explicitly if the two should differ.
- **`python -m irodori_openai_tts` accepts `--reload`** in addition to `--host` and
  `--port`, which runs uvicorn's autoreloader for development.
