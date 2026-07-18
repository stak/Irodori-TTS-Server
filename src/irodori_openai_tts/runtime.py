from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    default_runtime_device,
)

from .config import Settings

logger = logging.getLogger(__name__)


class RuntimeLoadTimeoutError(RuntimeError):
    pass


class RuntimeManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._runtime: InferenceRuntime | None = None
        self._checkpoint_path: str | None = None
        self._prewarm_status: str | None = None

    def get(self) -> InferenceRuntime:
        if self._runtime is not None:
            return self._runtime

        timeout = float(self.settings.model_load_timeout)
        acquired = self._lock.acquire(timeout=max(timeout, 0.0))
        if not acquired:
            raise RuntimeLoadTimeoutError(
                f"Model is still loading. Retry after a moment. timeout={timeout:.1f}s"
            )

        try:
            if self._runtime is None:
                logger.info("loading runtime")
                t0 = time.perf_counter()
                self._checkpoint_path = self._resolve_checkpoint_path()
                logger.info("checkpoint resolved: %s", self._checkpoint_path)
                self._runtime = InferenceRuntime.from_key(
                    RuntimeKey(
                        checkpoint=self._checkpoint_path,
                        model_device=self._resolve_device(self.settings.model_device),
                        codec_repo=str(self.settings.codec_repo),
                        model_precision=str(self.settings.model_precision),
                        codec_device=self._resolve_device(self.settings.codec_device),
                        codec_precision=str(self.settings.codec_precision),
                        codec_deterministic_encode=bool(self.settings.codec_deterministic_encode),
                        codec_deterministic_decode=bool(self.settings.codec_deterministic_decode),
                        compile_model=bool(self.settings.compile_model),
                        compile_dynamic=bool(self.settings.compile_dynamic),
                    )
                )
                elapsed = time.perf_counter() - t0
                logger.info("runtime loaded in %.2fs", elapsed)
            return self._runtime
        finally:
            self._lock.release()

    def prewarm(
        self,
        *,
        lora_adapter: str | None = None,
        lora_hot_swap: bool | None = None,
        max_seconds: float | None = None,
        num_steps: int | None = None,
    ) -> str:
        """
        Capture CUDA graphs (and warm the codec decoder / watermarker) so the
        first real request takes the fast replay path. Graphs are keyed by
        CFG scales and shapes, so prewarming uses the server's default
        sampling settings; requests that override CFG scales or supply
        reference audio still capture on first use.
        """
        runtime = self.get()
        settings = self.settings
        if lora_adapter is None:
            lora_adapter = settings.prewarm_lora_adapter
        if lora_adapter is not None and str(lora_adapter).strip() == "":
            lora_adapter = None
        status = runtime.prewarm_cuda_graphs(
            lora_adapter=lora_adapter,
            lora_hot_swap=(
                settings.default_lora_hot_swap if lora_hot_swap is None else bool(lora_hot_swap)
            ),
            max_seconds=float(settings.prewarm_max_seconds if max_seconds is None else max_seconds),
            num_steps=int(settings.default_num_steps if num_steps is None else num_steps),
            num_candidates=int(settings.default_num_candidates),
            cfg_guidance_mode=str(settings.default_cfg_guidance_mode),
            cfg_scale_text=float(settings.default_cfg_scale_text),
            cfg_scale_caption=float(settings.default_cfg_scale_text),
            cfg_scale_speaker=float(settings.default_cfg_scale_speaker),
            cfg_min_t=float(settings.default_cfg_min_t),
            cfg_max_t=float(settings.default_cfg_max_t),
            context_kv_cache=bool(settings.default_context_kv_cache),
            t_schedule_mode=str(settings.default_t_schedule_mode),
            sway_coeff=float(settings.default_sway_coeff),
            log_fn=lambda message: logger.info("irodori runtime: %s", message),
        )
        self._prewarm_status = status
        return status

    @property
    def prewarm_status(self) -> str | None:
        return self._prewarm_status

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def is_loading(self) -> bool:
        return self._runtime is None and self._lock.locked()

    def _resolve_checkpoint_path(self) -> str:
        if self.settings.checkpoint is not None and str(self.settings.checkpoint).strip() != "":
            path = Path(str(self.settings.checkpoint)).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            return str(path)

        repo_id = str(self.settings.hf_checkpoint).strip()
        if repo_id == "":
            raise ValueError("Set IRODORI_CHECKPOINT or IRODORI_HF_CHECKPOINT.")
        logger.info("downloading checkpoint from hf://%s/model.safetensors", repo_id)
        t0 = time.perf_counter()
        path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        elapsed = time.perf_counter() - t0
        logger.info("checkpoint download/cache lookup completed in %.2fs", elapsed)
        return path

    @staticmethod
    def _resolve_device(value: str) -> str:
        raw = str(value).strip().lower()
        if raw in {"", "auto"}:
            return default_runtime_device()
        return str(value)
