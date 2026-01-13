"""Message schemas for ZMQ communication."""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class AudioRequest:
    """Request message containing audio data for transcription."""
    
    request_id: str
    audio_format: Literal["wav", "flac"]
    sample_rate: int
    audio_data: bytes
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate the request data.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.request_id:
            return False, "request_id cannot be empty"
        
        if self.audio_format not in ["wav", "flac"]:
            return False, f"Invalid audio format: {self.audio_format}. Must be 'wav' or 'flac'"
        
        if self.sample_rate <= 0:
            return False, f"Invalid sample rate: {self.sample_rate}"
        
        if not self.audio_data:
            return False, "audio_data cannot be empty"
        
        return True, None


@dataclass
class TranscriptionResponse:
    """Response message containing transcription results or errors."""
    
    request_id: str
    status: Literal["success", "error"]
    text: str
    confidence: Optional[float] = None
    processing_time_ms: float = 0.0
    error_details: Optional[str] = None
    
    @classmethod
    def create_success(
        cls,
        request_id: str,
        text: str,
        processing_time_ms: float,
        confidence: Optional[float] = None
    ) -> "TranscriptionResponse":
        """Create a successful transcription response."""
        return cls(
            request_id=request_id,
            status="success",
            text=text,
            confidence=confidence,
            processing_time_ms=processing_time_ms
        )
    
    @classmethod
    def create_error(
        cls,
        request_id: str,
        error_message: str,
        processing_time_ms: float = 0.0
    ) -> "TranscriptionResponse":
        """Create an error response."""
        return cls(
            request_id=request_id,
            status="error",
            text="",
            processing_time_ms=processing_time_ms,
            error_details=error_message
        )
