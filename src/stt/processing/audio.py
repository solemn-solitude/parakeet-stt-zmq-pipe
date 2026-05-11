"""Audio processing and validation utilities."""
import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio validation and optional mono conversion."""

    def __init__(self, expected_sample_rate: int = 16000, convert_to_mono: bool = False):
        self.expected_sample_rate = expected_sample_rate
        self.convert_to_mono = convert_to_mono

        logger.info(
            "AudioProcessor initialized: sample_rate=%dHz, convert_to_mono=%s",
            expected_sample_rate,
            convert_to_mono,
        )

    def validate_and_process(
        self,
        audio_data: bytes,
        audio_format: str,
    ) -> tuple[bool, str | None, Path | None]:
        """Validate audio data and optionally convert to mono.

        Returns:
            Tuple of (is_valid, error_message, temp_file_path)
        """
        try:
            temp_path = self._write_temp_file(audio_data, audio_format)

            try:
                data, sample_rate = self._load_and_validate_audio(temp_path, audio_format)

                is_valid, error_msg = self._validate_sample_rate(sample_rate)
                if not is_valid:
                    temp_path.unlink()
                    return False, error_msg, None

                return self._handle_stereo_audio(data, sample_rate, temp_path, audio_format)

            except Exception:
                if temp_path.exists():
                    temp_path.unlink()
                raise

        except sf.LibsndfileError as e:
            error_msg = f"Failed to read audio file: {e}"
            logger.error(error_msg)
            return False, error_msg, None
        except Exception as e:
            error_msg = f"Unexpected error processing audio: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, None

    def _write_temp_file(self, audio_data: bytes, audio_format: str) -> Path:
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.write(audio_data)
        temp_file.close()
        return temp_path

    def _load_and_validate_audio(
        self,
        temp_path: Path,
        audio_format: str,
    ) -> tuple[np.ndarray, int]:
        data, sample_rate = sf.read(str(temp_path))

        duration_seconds = len(data) / sample_rate if sample_rate > 0 else 0
        num_channels = data.shape[1] if len(data.shape) > 1 else 1

        logger.info(
            "AUDIO FILE ANALYSIS - Path: %s, Sample Rate: %dHz, Channels: %d, "
            "Samples: %d, Duration: %.2fs, Shape: %s, Format: %s",
            temp_path,
            sample_rate,
            num_channels,
            len(data),
            duration_seconds,
            data.shape,
            audio_format,
        )

        return data, sample_rate

    def _validate_sample_rate(self, sample_rate: int) -> tuple[bool, str | None]:
        if sample_rate != self.expected_sample_rate:
            error_msg = (
                f"Invalid sample rate: expected {self.expected_sample_rate}Hz, "
                f"got {sample_rate}Hz"
            )
            logger.warning(error_msg)
            return False, error_msg

        return True, None

    def _handle_stereo_audio(
        self,
        data: np.ndarray,
        sample_rate: int,
        temp_path: Path,
        audio_format: str,
    ) -> tuple[bool, str | None, Path | None]:
        is_stereo = len(data.shape) == 2 and data.shape[1] == 2

        if not is_stereo:
            logger.debug("Audio is mono, no conversion needed")
            return True, None, temp_path

        if not self.convert_to_mono:
            error_msg = (
                "Audio is stereo but mono conversion is disabled. "
                "Expected mono audio (1 channel)."
            )
            logger.warning(error_msg)
            temp_path.unlink()
            return False, error_msg, None

        mono_path = self._convert_to_mono(data, sample_rate, audio_format)
        temp_path.unlink()

        return True, None, mono_path

    def _convert_to_mono(
        self,
        stereo_data: np.ndarray,
        sample_rate: int,
        audio_format: str,
    ) -> Path:
        logger.info("Converting stereo audio to mono")
        mono_data = np.mean(stereo_data, axis=1)

        mono_temp = tempfile.NamedTemporaryFile(suffix=f"_mono.{audio_format}", delete=False)
        mono_path = Path(mono_temp.name)
        mono_temp.close()

        sf.write(str(mono_path), mono_data, sample_rate)

        logger.debug("Mono conversion complete: %s", mono_path)
        return mono_path

    @staticmethod
    def cleanup_temp_file(file_path: Path | None) -> None:
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.debug("Cleaned up temp file: %s", file_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file %s: %s", file_path, e)
