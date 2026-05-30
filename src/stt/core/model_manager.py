"""Model manager with lazy loading and timeout-based deallocation."""
import gc
import logging
import threading
import time

import nemo.collections.asr as nemo_asr

from ainet.errors import classify_exception, mark_reported, report


logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the ASR model lifecycle with lazy loading and timeout."""

    def __init__(self, model_name: str, timeout_minutes: int = 10):
        self.model_name = model_name
        self.timeout_seconds = timeout_minutes * 60

        self._model: nemo_asr.models.ASRModel | None = None
        self._last_used_time: float = 0
        self._lock = threading.RLock()
        self._monitor_thread: threading.Thread | None = None
        self._running = False
        # Sticky: once load fails, subsequent get_model() calls fail fast
        # against this same error rather than retrying NeMo's from_pretrained.
        # Without this every incoming audio request would trigger another OOM
        # against the same VRAM pressure. A process restart clears it.
        self._init_error: BaseException | None = None

        logger.info(
            "ModelManager initialized: model=%s, timeout=%d minutes",
            model_name,
            timeout_minutes,
        )

    def start_monitoring(self) -> None:
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_timeout,
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Model timeout monitoring started")

    def stop_monitoring(self) -> None:
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Model timeout monitoring stopped")

    def get_model(self) -> nemo_asr.models.ASRModel:
        with self._lock:
            if self._init_error is not None:
                raise RuntimeError(
                    f"Model is in a permanently-failed state for this process: "
                    f"{self._init_error}. Restart the STT service to retry."
                ) from self._init_error
            if self._model is None:
                self._load_model()

            self._last_used_time = time.time()
            return self._model

    def _load_model(self) -> None:
        logger.info("Loading model: %s", self.model_name)
        start_time = time.time()

        try:
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )
            load_time = time.time() - start_time
            logger.info("Model loaded successfully in %.2f seconds", load_time)
            self._last_used_time = time.time()

        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_name, e, exc_info=True)
            # Report at the root cause, before the wrapping below — keeps the
            # exception type intact for the classifier (the wrapped RuntimeError
            # would still match via the message-substring path, but the
            # exc_type field in detail stays correct this way).
            kind, detail = classify_exception(e)
            detail["model_name"] = self.model_name
            report(
                service="stt",
                kind=kind,
                message=f"STT model failed to load: {e}",
                detail=detail,
                recoverable=True,
            )
            mark_reported(e)
            self._init_error = e
            wrapped = RuntimeError(f"Model loading failed: {e}")
            mark_reported(wrapped)
            raise wrapped from e

    def _deallocate_model(self) -> None:
        with self._lock:
            if self._model is not None:
                logger.info("Deallocating model due to timeout")
                self._model = None

                gc.collect()

                try:
                    import torch  # noqa: PLC0415 — optional CUDA dependency
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("CUDA cache cleared")
                except ImportError:
                    pass

                logger.info("Model deallocated successfully")

    def _monitor_timeout(self) -> None:
        check_interval = 60

        while self._running:
            time.sleep(check_interval)

            if not self._running:
                break

            with self._lock:
                if self._model is not None:
                    idle_time = time.time() - self._last_used_time

                    if idle_time >= self.timeout_seconds:
                        logger.info(
                            "Model idle for %.1f minutes, deallocating...",
                            idle_time / 60,
                        )
                        self._deallocate_model()

    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def force_reload(self) -> None:
        with self._lock:
            logger.info("Forcing model reload")
            self._deallocate_model()
            self._load_model()
