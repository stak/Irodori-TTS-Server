from __future__ import annotations

import threading

import pytest

from irodori_openai_tts import runtime as runtime_module
from irodori_openai_tts.config import Settings
from irodori_openai_tts.runtime import RuntimeLoadTimeoutError, RuntimeManager


def test_runtime_load_timeout_while_another_thread_is_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    settings = Settings(
        checkpoint=str(checkpoint),
        model_device="cpu",
        codec_device="cpu",
        model_load_timeout=0.05,
        _env_file=None,
    )
    manager = RuntimeManager(settings)
    started = threading.Event()
    release = threading.Event()
    loaded_runtime = object()
    errors: list[BaseException] = []

    def fake_from_key(_key):
        started.set()
        release.wait(timeout=2)
        return loaded_runtime

    monkeypatch.setattr(
        runtime_module.InferenceRuntime,
        "from_key",
        staticmethod(fake_from_key),
    )

    def load_runtime():
        try:
            assert manager.get() is loaded_runtime
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=load_runtime)
    thread.start()
    assert started.wait(timeout=1)

    with pytest.raises(RuntimeLoadTimeoutError):
        manager.get()

    release.set()
    thread.join(timeout=2)
    assert errors == []
    assert manager.is_loaded
    assert not manager.is_loading


class FakePrewarmRuntime:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def prewarm_cuda_graphs(self, **kwargs):
        self.calls.append(kwargs)
        return "prewarmed"


def test_prewarm_maps_server_defaults_to_runtime_arguments():
    settings = Settings(
        checkpoint="unused",
        prewarm_max_seconds=12.0,
        prewarm_lora_adapter="/adapters/base",
        default_lora_hot_swap=True,
        default_num_steps=24,
        default_num_candidates=2,
        default_cfg_guidance_mode="independent",
        default_cfg_scale_text=2.5,
        default_cfg_scale_speaker=4.5,
        default_cfg_min_t=0.4,
        default_cfg_max_t=0.9,
        default_context_kv_cache=False,
        default_t_schedule_mode="sway",
        default_sway_coeff=-0.5,
        _env_file=None,
    )
    manager = RuntimeManager(settings)
    runtime = FakePrewarmRuntime()
    manager._runtime = runtime

    status = manager.prewarm()

    assert status == "prewarmed"
    assert manager.prewarm_status == "prewarmed"
    call = runtime.calls[0]
    call.pop("log_fn")
    assert call == {
        "lora_adapter": "/adapters/base",
        "lora_hot_swap": True,
        "max_seconds": 12.0,
        "num_steps": 24,
        "num_candidates": 2,
        "cfg_guidance_mode": "independent",
        "cfg_scale_text": 2.5,
        "cfg_scale_caption": 2.5,
        "cfg_scale_speaker": 4.5,
        "cfg_min_t": 0.4,
        "cfg_max_t": 0.9,
        "context_kv_cache": False,
        "t_schedule_mode": "sway",
        "sway_coeff": -0.5,
    }


def test_prewarm_overrides_take_precedence_and_blank_adapter_is_none():
    settings = Settings(
        checkpoint="unused",
        prewarm_lora_adapter="",
        default_lora_hot_swap=True,
        _env_file=None,
    )
    manager = RuntimeManager(settings)
    runtime = FakePrewarmRuntime()
    manager._runtime = runtime

    manager.prewarm(lora_hot_swap=False, max_seconds=5.0, num_steps=8)

    call = runtime.calls[0]
    assert call["lora_adapter"] is None
    assert call["lora_hot_swap"] is False
    assert call["max_seconds"] == 5.0
    assert call["num_steps"] == 8


def test_runtime_resolves_local_checkpoint_path(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    manager = RuntimeManager(Settings(checkpoint=str(checkpoint), _env_file=None))

    assert manager._resolve_checkpoint_path() == str(checkpoint)


def test_runtime_rejects_missing_local_checkpoint(tmp_path):
    manager = RuntimeManager(
        Settings(checkpoint=str(tmp_path / "missing.safetensors"), _env_file=None)
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        manager._resolve_checkpoint_path()


def test_runtime_downloads_hf_checkpoint_when_local_checkpoint_is_unset(monkeypatch):
    manager = RuntimeManager(Settings(hf_checkpoint="owner/repo", _env_file=None))

    def fake_download(*, repo_id, filename):
        assert repo_id == "owner/repo"
        assert filename == "model.safetensors"
        return "/cache/model.safetensors"

    monkeypatch.setattr(runtime_module, "hf_hub_download", fake_download)

    assert manager._resolve_checkpoint_path() == "/cache/model.safetensors"
