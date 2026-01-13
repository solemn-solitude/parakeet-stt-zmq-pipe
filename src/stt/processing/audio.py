"""Audio processing and validation utilities."""
import io
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
            # Write audio data to temporary file for processing
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f".{audio_format}",
                delete=False
            )
            temp_path = Path(temp_file.name)
            
            try:
                temp_file.write(audio_data)
                temp_file.close()
                
                # Read audio file to validate
                data, sample_rate = sf.read(str(temp_path))
                
                logger.debug(
                    f"Audio loaded: shape={data.shape}, sample_rate={sample_rate}Hz, "
                    f"format={audio_format}"
                )
                
                # Validate sample rate
                if sample_rate != self.expected_sample_rate:
                    error_msg = (
                        f"Invalid sample rate: expected {self.expected_sample_rate}Hz, "
                        f"got {sample_rate}Hz"
                    )
                    logger.warning(error_msg)
                    temp_path.unlink()  # Clean up on error
                    return False, error_msg, None
                
                # Check if audio is stereo
                is_stereo = len(data.shape) == 2 and data.shape[1] == 2
                
                if is_stereo:
                    if not self.convert_to_mono:
                        error_msg = (
                            "Audio is stereo but mono conversion is disabled. "
                            "Expected mono audio (1 channel)."
                        )
                        logger.warning(error_msg)
                        temp_path.unlink()  # Clean up on error
                        return False, error_msg, None
                    
                    # Convert stereo to mono
                    logger.info("Converting stereo audio to mono")
                    mono_data = np.mean(data, axis=1)
                    
                    # Create new temp file for mono audio
                    mono_temp = tempfile.NamedTemporaryFile(
                        suffix=f"_mono.{audio_format}",
                        delete=False
                    )
                    mono_path = Path(mono_temp.name)
                    mono_temp.close()
                    
                    # Write mono audio
                    sf.write(str(mono_path), mono_data, sample_rate)
                    
                    # Clean up original stereo file
                    temp_path.unlink()
                    
                    logger.debug(f"Mono conversion complete: {mono_path}")
                    return True, None, mono_path
                
                # Audio is already mono
                logger.debug("Audio is mono, no conversion needed")
                return True, None, temp_path
                
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
