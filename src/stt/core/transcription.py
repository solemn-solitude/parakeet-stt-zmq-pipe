"""Core transcription functionality."""
import logging
import time
from pathlib import Path
from typing import Optional

from .model_manager import ModelManager


logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """Handles audio transcription using the ASR model."""
    
    def __init__(self, model_manager: ModelManager):
        """Initialize the transcription engine.
        
        Args:
            model_manager: The model manager instance
        """
        self.model_manager = model_manager
        logger.info("TranscriptionEngine initialized")
    
    def transcribe(
        self,
        audio_file_path: Path
    ) -> tuple[str, float, Optional[float]]:
        """Transcribe an audio file.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Tuple of (transcription_text, processing_time_ms, confidence)
            
        Raises:
            RuntimeError: If transcription fails
        """
        start_time = time.time()
        
        try:
            # Get the model (lazy loads if needed)
            model = self.model_manager.get_model()
            
            logger.debug(f"Transcribing audio file: {audio_file_path}")
            
            # Perform transcription
            # NeMo's transcribe method returns a list of results
            results = model.transcribe([str(audio_file_path)])
            
            # Extract text from results
            if not results or len(results) == 0:
                raise RuntimeError("Transcription returned empty results")
            
            # Results can be in different formats depending on the model
            # Handle both list of strings and list of Hypothesis objects
            result = results[0]
            
            if hasattr(result, 'text'):
                # Hypothesis object
                text = result.text
                confidence = getattr(result, 'score', None)
            elif isinstance(result, str):
                # Direct string result
                text = result
                confidence = None
            else:
                # Try to get text attribute or convert to string
                text = str(result)
                confidence = None
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                f"Transcription complete: length={len(text)} chars, "
                f"time={processing_time_ms:.2f}ms"
            )
            logger.debug(f"Transcription result: {text[:100]}...")
            
            return text, processing_time_ms, confidence
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Transcription failed after {processing_time_ms:.2f}ms: {e}",
                exc_info=True
            )
            raise RuntimeError(f"Transcription failed: {e}") from e
