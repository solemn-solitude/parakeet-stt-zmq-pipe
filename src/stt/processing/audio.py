"""Audio processing and validation utilities."""
import logging
import tempfile
from pathlib import Path
from typing import Tuple, Optional

import soundfile as sf
import numpy as np


logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio validation and optional mono conversion."""
    
    def __init__(self, expected_sample_rate: int = 16000, convert_to_mono: bool = False):
        """Initialize the audio processor.
        
        Args:
            expected_sample_rate: Expected sample rate in Hz (default: 16000)
            convert_to_mono: Whether to convert stereo audio to mono
        """
        self.expected_sample_rate = expected_sample_rate
        self.convert_to_mono = convert_to_mono
        
        logger.info(
            f"AudioProcessor initialized: sample_rate={expected_sample_rate}Hz, "
            f"convert_to_mono={convert_to_mono}"
        )
    
    def validate_and_process(
        self, 
        audio_data: bytes, 
        audio_format: str
    ) -> Tuple[bool, Optional[str], Optional[Path]]:
        """Validate audio data and optionally convert to mono.
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format ('wav' or 'flac')
            
        Returns:
            Tuple of (is_valid, error_message, temp_file_path)
            - is_valid: Whether the audio passed validation
            - error_message: Error description if validation failed
            - temp_file_path: Path to temporary audio file (caller should clean up)
        """
        try:
            # Write audio data to temporary file
            temp_path = self._write_temp_file(audio_data, audio_format)
            
            try:
                # Load and validate audio
                data, sample_rate = self._load_and_validate_audio(temp_path, audio_format)
                
                # Validate sample rate
                is_valid, error_msg = self._validate_sample_rate(sample_rate)
                if not is_valid:
                    temp_path.unlink()
                    return False, error_msg, None
                
                # Handle stereo audio if needed
                return self._handle_stereo_audio(data, sample_rate, temp_path, audio_format)
                
            except Exception as e:
                # Clean up temp file on error
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
        """Write audio data to a temporary file.
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format extension
            
        Returns:
            Path to the temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(
            suffix=f".{audio_format}",
            delete=False
        )
        temp_path = Path(temp_file.name)
        temp_file.write(audio_data)
        temp_file.close()
        return temp_path
    
    def _load_and_validate_audio(
        self, 
        temp_path: Path, 
        audio_format: str
    ) -> Tuple[np.ndarray, int]:
        """Load audio file and return data and sample rate.
        
        Args:
            temp_path: Path to the temporary audio file
            audio_format: Audio format for logging
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        data, sample_rate = sf.read(str(temp_path))
        
        duration_seconds = len(data) / sample_rate if sample_rate > 0 else 0
        num_channels = data.shape[1] if len(data.shape) > 1 else 1
        
        logger.info(
            f"AUDIO FILE ANALYSIS - Path: {temp_path}, "
            f"Sample Rate: {sample_rate}Hz, "
            f"Channels: {num_channels}, "
            f"Samples: {len(data)}, "
            f"Duration: {duration_seconds:.2f}s, "
            f"Shape: {data.shape}, "
            f"Format: {audio_format}"
        )
        
        return data, sample_rate
    
    def _validate_sample_rate(self, sample_rate: int) -> Tuple[bool, Optional[str]]:
        """Validate that the sample rate matches expected value.
        
        Args:
            sample_rate: The actual sample rate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
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
        audio_format: str
    ) -> Tuple[bool, Optional[str], Optional[Path]]:
        """Handle stereo audio - either reject or convert to mono.
        
        Args:
            data: Audio data array
            sample_rate: Sample rate in Hz
            temp_path: Path to the original temporary file
            audio_format: Audio format extension
            
        Returns:
            Tuple of (is_valid, error_message, temp_file_path)
        """
        is_stereo = len(data.shape) == 2 and data.shape[1] == 2
        
        if not is_stereo:
            # Audio is already mono
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
        
        # Convert stereo to mono
        mono_path = self._convert_to_mono(data, sample_rate, audio_format)
        temp_path.unlink()  # Clean up original stereo file
        
        return True, None, mono_path
    
    def _convert_to_mono(
        self, 
        stereo_data: np.ndarray, 
        sample_rate: int,
        audio_format: str
    ) -> Path:
        """Convert stereo audio to mono and write to new temp file.
        
        Args:
            stereo_data: Stereo audio data array
            sample_rate: Sample rate in Hz
            audio_format: Audio format extension
            
        Returns:
            Path to the mono audio temporary file
        """
        logger.info("Converting stereo audio to mono")
        mono_data = np.mean(stereo_data, axis=1)
        
        # Create new temp file for mono audio
        mono_temp = tempfile.NamedTemporaryFile(
            suffix=f"_mono.{audio_format}",
            delete=False
        )
        mono_path = Path(mono_temp.name)
        mono_temp.close()
        
        # Write mono audio
        sf.write(str(mono_path), mono_data, sample_rate)
        
        logger.debug(f"Mono conversion complete: {mono_path}")
        return mono_path
    
    @staticmethod
    def cleanup_temp_file(file_path: Optional[Path]) -> None:
        """Clean up a temporary file.
        
        Args:
            file_path: Path to the temporary file to delete
        """
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {file_path}: {e}")
