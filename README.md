# Speech-to-Text (STT) Service

A professional, production-ready Speech-to-Text service using NVIDIA NeMo's Parakeet TDT 0.6B v2 model with ZMQ messaging for pipeline integration.

## Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  Audio Service  │         │   STT Service   │         │   LLM Service   │
│    (DEALER)     │────────▶│  ROUTER→DEALER  │────────▶│    (ROUTER)     │
└─────────────────┘         └─────────────────┘         └─────────────────┘
     Upstream                 This Service                 Downstream
```

### Components

```
src/stt/
├── cli.py                    # Click CLI commands
├── config.py                 # Configuration dataclasses
├── service.py                # Main service orchestration & event loop
├── core/
│   ├── model_manager.py      # Model lifecycle (lazy load, timeout, deallocation)
│   └── transcription.py      # Core transcription logic
├── messaging/
│   ├── schemas.py            # Message dataclasses (AudioRequest, TranscriptionResponse)
│   ├── serialization.py      # msgpack encode/decode
│   └── zmq_handler.py        # ZMQ ROUTER (input) + DEALER (output) management
├── processing/
│   └── audio.py              # Audio validation, optional mono conversion
└── utils/
    └── logging.py            # Rotating file handler with size & time-based rotation
```

## Features

- ✅ **ZMQ Messaging**: ROUTER (input) + DEALER (output) pattern for pipeline integration
- ✅ **Lazy Model Loading**: Model loads on first request, not at startup
- ✅ **Memory Management**: Automatic model deallocation after configurable timeout (default: 10 minutes)
- ✅ **Audio Validation**: Validates format (wav/flac), sample rate (16kHz), channel count
- ✅ **Optional Mono Conversion**: Flag-controlled stereo→mono conversion
- ✅ **Comprehensive Logging**: Rotating log files (10MB max, weekly rotation, hourly flush)
- ✅ **Proper Error Handling**: All errors caught, logged, and sent as structured responses
- ✅ **Thread-Safe**: Thread-safe model access and timeout monitoring
- ✅ **Signal Handling**: Graceful shutdown on SIGINT/SIGTERM

## Installation

```bash
# Dependencies are managed with uv
uv sync
```

## Usage

### Start the Service

```bash
# Basic usage (defaults)
uv run python main.py start

# Custom configuration
uv run python main.py start \
  --input-address tcp://*:5555 \
  --output-address tcp://localhost:5556 \
  --timeout 15 \
  --convert-to-mono \
  --log-file /var/log/stt.log \
  --log-level INFO
```

### CLI Options

```
--input-address     ZMQ ROUTER input address (default: tcp://*:5555)
--output-address    ZMQ DEALER output address (default: tcp://localhost:5556)
--timeout           Model idle timeout in minutes (default: 10)
--convert-to-mono   Enable stereo→mono conversion (flag, default: disabled)
--log-file          Log file path (default: stt.log)
--log-level         Logging level: DEBUG|INFO|WARNING|ERROR|CRITICAL (default: WARNING)
```

### Version Info

```bash
uv run python main.py version
```

## Message Protocol

### Input: AudioRequest

```python
@dataclass
class AudioRequest:
    request_id: str           # Unique request identifier
    audio_format: str         # "wav" or "flac"
    sample_rate: int          # Sample rate in Hz (expected: 16000)
    audio_data: bytes         # Raw audio bytes
```

### Output: TranscriptionResponse

```python
@dataclass
class TranscriptionResponse:
    request_id: str               # Matching request ID
    status: str                   # "success" or "error"
    text: str                     # Transcription text (empty on error)
    confidence: Optional[float]   # Confidence score (if available)
    processing_time_ms: float     # Processing time in milliseconds
    error_details: Optional[str]  # Error message (if status="error")
```

Messages are serialized using **msgpack** for efficiency.

## Audio Requirements

- **Format**: WAV or FLAC
- **Sample Rate**: 16kHz (validated)
- **Channels**: Mono (1 channel)
  - Stereo audio requires `--convert-to-mono` flag
  - Without flag, stereo audio will be rejected

## Testing

A test client is provided:

```bash
# In terminal 1: Start the service
uv run python main.py start --convert-to-mono

# In terminal 2: Run test client
uv run python test_client.py
```

The test client sends `sora_sample.wav` and prints the transcription response.

## Example Client Code

```python
import zmq
from src.stt.messaging.schemas import AudioRequest
from src.stt.messaging.serialization import (
    serialize_audio_request,
    deserialize_transcription_response,
)

# Connect to service
context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.connect("tcp://localhost:5555")

# Load audio
with open("audio.wav", "rb") as f:
    audio_data = f.read()

# Create request
request = AudioRequest(
    request_id="unique-id-123",
    audio_format="wav",
    sample_rate=16000,
    audio_data=audio_data,
)

# Send
socket.send(serialize_audio_request(request))

# Receive
response_bytes = socket.recv()
response = deserialize_transcription_response(response_bytes)

print(f"Status: {response.status}")
print(f"Text: {response.text}")
```

## Model Information

- **Model**: NVIDIA NeMo Parakeet TDT 0.6B v2
- **Type**: Transducer-based ASR (RNN-T with TDT alignment)
- **Speed**: Optimized for low-latency transcription
- **Language**: English

## Logging

Logs are written to file with:
- **Default Level**: WARNING (errors + warnings only)
- **Rotation**: Size-based (10MB) + weekly
- **Flush**: Hourly automatic flush
- **Format**: `TIMESTAMP - MODULE - LEVEL - MESSAGE`

Check logs for detailed service operation and debugging.

## Production Considerations

1. **Memory**: Model uses ~2GB GPU memory when loaded
2. **Timeout**: Adjust `--timeout` based on expected request frequency
3. **Logging**: Use `--log-level WARNING` or `ERROR` in production
4. **Network**: Ensure ZMQ ports are accessible between services
5. **Error Handling**: All errors are caught and logged; clients receive error responses

## Troubleshooting

### Service won't start
- Check if ports are already in use
- Verify dependencies installed: `uv sync`

### Stereo audio rejected
- Enable mono conversion: `--convert-to-mono`
- Or convert audio upstream for better performance

### Model loading slow
- First request loads model (5-10 seconds)
- Subsequent requests are fast
- Model stays loaded for `--timeout` minutes after last use

### High memory usage
- Model is loaded; this is expected
- Reduce `--timeout` for faster deallocation
- Memory freed automatically after timeout

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
