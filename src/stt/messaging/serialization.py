"""Serialization utilities for msgpack encoding/decoding of message schemas."""
import msgpack
from typing import Union

from .schemas import AudioRequest, TranscriptionResponse


def serialize_audio_request(request: AudioRequest) -> bytes:
    """Serialize an AudioRequest to msgpack bytes.
    
    Args:
        request: The AudioRequest to serialize
        
    Returns:
        Msgpack-encoded bytes
    """
    data = {
        "request_id": request.request_id,
        "audio_format": request.audio_format,
        "sample_rate": request.sample_rate,
        "audio_data": request.audio_data,
    }
    return msgpack.packb(data, use_bin_type=True)


def deserialize_audio_request(data: bytes) -> AudioRequest:
    """Deserialize msgpack bytes to an AudioRequest.
    
    Args:
        data: Msgpack-encoded bytes
        
    Returns:
        Deserialized AudioRequest object
        
    Raises:
        ValueError: If deserialization fails
    """
    try:
        obj = msgpack.unpackb(data, raw=False)
        return AudioRequest(
            request_id=obj["request_id"],
            audio_format=obj["audio_format"],
            sample_rate=obj["sample_rate"],
            audio_data=obj["audio_data"],
        )
    except (KeyError, msgpack.exceptions.UnpackException) as e:
        raise ValueError(f"Failed to deserialize AudioRequest: {e}") from e


def serialize_transcription_response(response: TranscriptionResponse) -> bytes:
    """Serialize a TranscriptionResponse to msgpack bytes.
    
    Args:
        response: The TranscriptionResponse to serialize
        
    Returns:
        Msgpack-encoded bytes
    """
    data = {
        "request_id": response.request_id,
        "status": response.status,
        "text": response.text,
        "confidence": response.confidence,
        "processing_time_ms": response.processing_time_ms,
        "error_details": response.error_details,
    }
    return msgpack.packb(data, use_bin_type=True)


def deserialize_transcription_response(data: bytes) -> TranscriptionResponse:
    """Deserialize msgpack bytes to a TranscriptionResponse.
    
    Args:
        data: Msgpack-encoded bytes
        
    Returns:
        Deserialized TranscriptionResponse object
        
    Raises:
        ValueError: If deserialization fails
    """
    try:
        obj = msgpack.unpackb(data, raw=False)
        return TranscriptionResponse(
            request_id=obj["request_id"],
            status=obj["status"],
            text=obj["text"],
            confidence=obj.get("confidence"),
            processing_time_ms=obj.get("processing_time_ms", 0.0),
            error_details=obj.get("error_details"),
        )
    except (KeyError, msgpack.exceptions.UnpackException) as e:
        raise ValueError(f"Failed to deserialize TranscriptionResponse: {e}") from e
