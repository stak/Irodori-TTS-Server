from __future__ import annotations

import pytest

from irodori_openai_tts import app as main


@pytest.fixture(autouse=True)
def restore_app_globals(monkeypatch, tmp_path):
    monkeypatch.delenv("IRODORI_PERF_PROFILE", raising=False)
    monkeypatch.setattr(main.settings, "api_key", None)
    monkeypatch.setattr(main.settings, "runtime_profile", "recommended")
    monkeypatch.setattr(main.settings, "voices_dir", tmp_path)
    monkeypatch.setattr(main.settings, "voice_aliases_file", None)
    monkeypatch.setattr(main.settings, "default_voice", None)
    monkeypatch.setattr(main.settings, "allow_no_ref_voice", True)
    monkeypatch.setattr(main.settings, "default_chunking_enabled", True)
    monkeypatch.setattr(main.settings, "default_chunk_min_chars", 80)
    monkeypatch.setattr(main.settings, "default_first_sentence_chunk_min_chars", None)
    monkeypatch.setattr(main.settings, "default_lora_hot_swap", False)
    monkeypatch.setattr(main.settings, "default_apply_watermark", True)
    monkeypatch.setattr(main.settings, "default_num_candidates", 1)
    monkeypatch.setattr(main.settings, "max_num_candidates", 8)
    monkeypatch.setattr(main.settings, "mp3_bitrate_mode", "VARIABLE")
    monkeypatch.setattr(main.settings, "mp3_compression_level", 0.0)
    monkeypatch.setattr(main.settings, "preload", False)
    monkeypatch.setattr(main.settings, "max_concurrent_synthesis", 1)
    monkeypatch.setattr(main.settings, "synthesis_wait_timeout", 300.0)
    monkeypatch.setattr(main, "_synthesis_semaphore", None)
    monkeypatch.setattr(main, "_synthesis_semaphore_limit", None)
    yield
