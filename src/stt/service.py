"""Main STT service orchestration and event loop."""
import logging
import signal
from pathlib import Path

from stt.config import STTConfig
from stt.core.model_manager import ModelManager
from stt.core.transcription import TranscriptionEngine
from stt.messaging.schemas import TranscriptionResponse
from stt.messaging.zmq_handler import ZMQHandler
from stt.processing.audio import AudioProcessor
from stt.utils.logging import PeriodicFlusher


logger = logging.getLogger(__name__)


class STTService:
    """Main service that orchestrates all components."""

    def __init__(self, config: STTConfig):
        self.config = config
        self.running = False

        self.model_manager = ModelManager(
            model_name=config.model_name,
            timeout_minutes=config.model_timeout_minutes,
        )

        self.transcription_engine = TranscriptionEngine(
            model_manager=self.model_manager,
        )

        self.audio_processor = AudioProcessor(
            expected_sample_rate=config.expected_sample_rate,
            convert_to_mono=config.convert_to_mono,
        )

        self.zmq_handler = ZMQHandler(
            input_address=config.input_address,
            output_address=config.output_address,
        )

        self.log_flusher = PeriodicFlusher(interval_seconds=3600)

        logger.info("STTService initialized")

    def setup(self) -> None:
        logger.info("Setting up STT service...")

        self.zmq_handler.setup()
        self.model_manager.start_monitoring()
        self.log_flusher.start()

        logger.info("STT service setup complete")

    def run(self) -> None:
        self.setup()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True
        logger.info("STT service started, waiting for requests...")

        try:
            while self.running:
                self._process_one_request()
        except Exception as e:
            logger.error("Fatal error in service loop: %s", e, exc_info=True)
        finally:
            self.cleanup()

    def _process_one_request(self) -> None:
        temp_file: Path | None = None

        try:
            result = self.zmq_handler.receive_request(timeout_ms=100)

            if result is None:
                return

            identity, request = result
            logger.info("Processing request %s", request.request_id)

            is_valid, error_msg = request.validate()
            if not is_valid:
                self._send_error_response(request, "Request validation failed", error_msg)
                return

            is_valid, error_msg, temp_file = self.audio_processor.validate_and_process(
                audio_data=request.audio_data,
                audio_format=request.audio_format,
            )

            if not is_valid:
                self._send_error_response(request, "Audio validation failed", error_msg)
                return

            self.zmq_handler.send_response(self._try_transcribe_audio(temp_file, request))

        except ValueError as e:
            logger.error("Failed to deserialize request: %s", e)

        except Exception as e:
            logger.error("Unexpected error processing request: %s", e, exc_info=True)

        finally:
            if temp_file is not None:
                AudioProcessor.cleanup_temp_file(temp_file)

    def _send_error_response(self, request, error_type: str, error_detail: str) -> None:
        logger.warning(
            "%s for request %s: %s", error_type, request.request_id, error_detail
        )
        response = TranscriptionResponse.create_error(
            request_id=request.request_id,
            error_message=f"{error_type}: {error_detail}",
        )
        self.zmq_handler.send_response(response)

    def _signal_handler(self, signum: int, frame: object) -> None:
        logger.info("Received signal %d, shutting down...", signum)
        self.running = False

    def cleanup(self) -> None:
        logger.info("Cleaning up STT service...")

        try:
            self.model_manager.stop_monitoring()
            self.log_flusher.stop()
            self.zmq_handler.cleanup()

        except Exception as e:
            logger.error("Error during cleanup: %s", e, exc_info=True)

        logger.info("STT service cleanup complete")

    def _try_transcribe_audio(self, temp_file: Path, request) -> TranscriptionResponse:
        try:
            text, processing_time_ms, confidence = self.transcription_engine.transcribe(
                audio_file_path=temp_file,
            )

            logger.info(
                "Request %s completed successfully in %.2fms",
                request.request_id,
                processing_time_ms,
            )

            return TranscriptionResponse.create_success(
                request_id=request.request_id,
                text=text,
                processing_time_ms=processing_time_ms,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(
                "Transcription failed for request %s: %s",
                request.request_id,
                e,
                exc_info=True,
            )
            return TranscriptionResponse.create_error(
                request_id=request.request_id,
                error_message=f"Transcription failed: {e}",
            )
