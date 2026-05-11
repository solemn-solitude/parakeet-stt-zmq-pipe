"""Message schemas for ZMQ communication."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class AudioRequest:
    """Request message containing audio data for transcription."""

    request_id: str
    audio_format: Literal["wav", "flac"]
    sample_rate: int
    audio_data: bytes

    def validate(self) -> tuple[bool, str | None]:
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
    confidence: float | None = None
    processing_time_ms: float = 0.0
    error_details: str | None = None

    @classmethod
    def create_success(
        cls,
        request_id: str,
        text: str,
        processing_time_ms: float,
        confidence: float | None = None,
    ) -> TranscriptionResponse:
        return cls(
            request_id=request_id,
            status="success",
            text=text,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )

    @classmethod
    def create_error(
        cls,
        request_id: str,
        error_message: str,
        processing_time_ms: float = 0.0,
    ) -> TranscriptionResponse:
        return cls(
            request_id=request_id,
            status="error",
            text="",
            processing_time_ms=processing_time_ms,
            error_details=error_message,
        )
