"""Core transcription functionality."""
import logging
import time
from pathlib import Path

from stt.core.model_manager import ModelManager


logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """Handles audio transcription using the ASR model."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        logger.info("TranscriptionEngine initialized")

    def transcribe(self, audio_file_path: Path) -> tuple[str, float, float | None]:
        """Transcribe an audio file.

        Returns:
            Tuple of (transcription_text, processing_time_ms, confidence)

        Raises:
            RuntimeError: If transcription fails
        """
        start_time = time.time()

        try:
            model = self.model_manager.get_model()

            logger.debug("Transcribing audio file: %s", audio_file_path)

            results = model.transcribe([str(audio_file_path)])

            if not results or len(results) == 0:
                raise RuntimeError("Transcription returned empty results")

            result = results[0]

            if hasattr(result, "text"):
                text = result.text
                confidence = getattr(result, "score", None)
            elif isinstance(result, str):
                text = result
                confidence = None
            else:
                text = str(result)
                confidence = None

            processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "Transcription complete: length=%d chars, time=%.2fms",
                len(text),
                processing_time_ms,
            )
            logger.debug("Transcription result: %s...", text[:100])

            return text, processing_time_ms, confidence

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(
                "Transcription failed after %.2fms: %s",
                processing_time_ms,
                e,
                exc_info=True,
            )
            raise RuntimeError(f"Transcription failed: {e}") from e
