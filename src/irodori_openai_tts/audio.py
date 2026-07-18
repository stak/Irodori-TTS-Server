from __future__ import annotations

import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf
import torch
import torchaudio

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def normalize_response_format(value: str | None, *, default: str) -> str:
    fmt = (default if value is None else str(value)).strip().lower()
    if fmt not in CONTENT_TYPES:
        allowed = ", ".join(sorted(CONTENT_TYPES))
        raise ValueError(f"Unsupported response_format={value!r}. Expected one of: {allowed}.")
    return fmt


MP3_BITRATE_MODES = {"CONSTANT", "AVERAGE", "VARIABLE"}


def encode_audio(
    audio: torch.Tensor,
    sample_rate: int,
    response_format: str,
    *,
    mp3_bitrate_mode: str = "VARIABLE",
    mp3_compression_level: float = 0.0,
) -> bytes:
    fmt = normalize_response_format(response_format, default="mp3")
    wav = audio.detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.ndim != 2:
        raise ValueError(f"Expected audio shape (channels, samples), got {tuple(wav.shape)}")
    wav = wav.clamp(-1.0, 1.0).contiguous()

    if fmt == "pcm":
        pcm = (wav.squeeze(0).numpy() * 32767.0).astype("<i2", copy=False)
        return pcm.tobytes()

    if fmt in {"wav", "flac", "mp3", "opus"}:
        encoder_kwargs = {}
        if fmt == "mp3":
            encoder_kwargs = _mp3_encoder_kwargs(mp3_bitrate_mode, mp3_compression_level)
        try:
            return _encode_with_soundfile(wav, int(sample_rate), fmt, **encoder_kwargs)
        except Exception as exc:
            if fmt in {"wav", "flac"}:
                raise
            soundfile_exc = exc
    else:
        soundfile_exc = None

    try:
        return _encode_with_torchaudio(wav, int(sample_rate), fmt)
    except Exception as torchaudio_exc:
        try:
            return _encode_with_ffmpeg(wav, int(sample_rate), fmt)
        except Exception as ffmpeg_exc:
            details = [f"torchaudio: {torchaudio_exc}"]
            if soundfile_exc is not None:
                details.insert(0, f"soundfile: {soundfile_exc}")
            details.append(f"ffmpeg: {ffmpeg_exc}")
            raise RuntimeError(
                f"Failed to encode audio as {fmt}. Install soundfile with MP3/Opus support, "
                "FFmpeg-enabled torchaudio, or ffmpeg in PATH; or request "
                "response_format='wav'/'flac'/'pcm'. "
                f"Encoder errors: {'; '.join(details)}"
            ) from ffmpeg_exc


def _mp3_encoder_kwargs(bitrate_mode: str, compression_level: float) -> dict:
    mode = str(bitrate_mode).strip().upper()
    if mode not in MP3_BITRATE_MODES:
        allowed = ", ".join(sorted(MP3_BITRATE_MODES))
        raise ValueError(f"mp3_bitrate_mode must be one of: {allowed} (got {bitrate_mode!r})")
    level = float(compression_level)
    # libsndfile accepts [0.0, 1.0) — 0.0 is the highest quality; for CBR the
    # level selects the bitrate (0.0 = 320k), for VBR the LAME -V quality.
    if not 0.0 <= level < 1.0:
        raise ValueError(
            f"mp3_compression_level must be within [0.0, 1.0) (got {compression_level!r})"
        )
    return {"bitrate_mode": mode, "compression_level": level}


def _soundfile_format(fmt: str) -> tuple[str, str | None]:
    if fmt == "mp3":
        return "MP3", "MPEG_LAYER_III"
    if fmt == "opus":
        return "OGG", "OPUS"
    return fmt.upper(), None


def _encode_with_soundfile(
    wav: torch.Tensor,
    sample_rate: int,
    fmt: str,
    **encoder_kwargs,
) -> bytes:
    audio = wav.transpose(0, 1).numpy()
    sf_format, subtype = _soundfile_format(fmt)
    buffer = BytesIO()
    sf.write(
        buffer,
        audio,
        sample_rate,
        format=sf_format,
        subtype=subtype,
        **encoder_kwargs,
    )
    return buffer.getvalue()


def _torchaudio_format(fmt: str) -> str:
    if fmt == "opus":
        return "ogg"
    if fmt == "aac":
        return "adts"
    return fmt


def _torchaudio_suffix(fmt: str) -> str:
    if fmt == "opus":
        return ".opus"
    if fmt == "aac":
        return ".aac"
    return f".{fmt}"


def _encode_with_torchaudio(wav: torch.Tensor, sample_rate: int, fmt: str) -> bytes:
    with TemporaryDirectory() as directory:
        path = Path(directory) / f"speech{_torchaudio_suffix(fmt)}"
        torchaudio.save(
            str(path),
            wav,
            sample_rate,
            format=_torchaudio_format(fmt),
        )
        return path.read_bytes()


def _ffmpeg_format(fmt: str) -> str:
    if fmt == "aac":
        return "adts"
    if fmt == "opus":
        return "ogg"
    return fmt


def _ffmpeg_codec(fmt: str) -> str:
    if fmt == "mp3":
        return "libmp3lame"
    if fmt == "opus":
        return "libopus"
    if fmt == "aac":
        return "aac"
    return fmt


def _encode_with_ffmpeg(wav: torch.Tensor, sample_rate: int, fmt: str) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable was not found in PATH")

    with TemporaryDirectory() as directory:
        source = Path(directory) / "speech.wav"
        target = Path(directory) / f"speech{_torchaudio_suffix(fmt)}"
        sf.write(
            source,
            wav.transpose(0, 1).numpy(),
            sample_rate,
            format="WAV",
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-codec:a",
            _ffmpeg_codec(fmt),
            "-f",
            _ffmpeg_format(fmt),
            str(target),
        ]
        subprocess.run(command, check=True, capture_output=True)
        return target.read_bytes()
