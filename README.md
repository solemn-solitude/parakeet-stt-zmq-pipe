# Parakeet STT ZMQ Pipe

Speech-to-text middleware using NeMo Parakeet TDT 0.6B v2 with ZMQ messaging.

## Architecture

```
Audio Service (DEALER) → [STT] → LLM/RAG Service (ROUTER)
```

- **Input**: ROUTER socket, receives audio bytes
- **Output**: DEALER socket, sends transcription text
- **Model**: Lazy-loaded, auto-deallocated after timeout

## Installation

```bash
uv sync
```

## Usage

```bash
# Start with defaults
uv run python main.py start

# Using environment variable
export STT_INPUT_ADDRESS="tcp://*:5555"
uv run python main.py start

# Custom config (CLI overrides env var)
uv run python main.py start \
  --input-address tcp://*:5555 \
  --output-address tcp://localhost:5556 \
  --timeout 10 \
  --convert-to-mono \
  --log-level INFO
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input-address` | `tcp://localhost:20499` | ROUTER bind address (overrides `STT_INPUT_ADDRESS` env var) |
| `--output-address` | `tcp://localhost:5556` | DEALER connect address |
| `--timeout` | `10` | Model idle timeout (minutes) |
| `--convert-to-mono` | `false` | Enable stereo→mono conversion |
| `--log-file` | `stt.log` | Log file path |
| `--log-level` | `WARNING` | DEBUG\|INFO\|WARNING\|ERROR\|CRITICAL |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `STT_INPUT_ADDRESS` | ROUTER bind address (default: `tcp://localhost:20499`, overridden by `--input-address` CLI flag) |

## Message Protocol

Messages are msgpack-encoded.

**Input (AudioRequest):**
```python
{
    "request_id": str,
    "audio_format": "wav" | "flac",
    "sample_rate": int,
    "audio_data": bytes
}
```

**Output (TranscriptionResponse):**
```python
{
    "request_id": str,
    "status": "success" | "error",
    "text": str,
    "confidence": float | null,
    "processing_time_ms": float,
    "error_details": str | null
}
```

## Requirements

- Audio: 16kHz, mono, WAV or FLAC
- Stereo audio requires `--convert-to-mono` flag

## Client Library (Downstream Services)

For LLM/RAG services consuming transcriptions:

```python
from src.stt.client import STTClient

def handle_transcription(response):
    if response.status == "success":
        print(f"Transcription: {response.text}")
        # Process with LLM, do RAG lookup, etc.
    else:
        print(f"Error: {response.error_details}")

# Simple callback-based approach
with STTClient(bind_address="tcp://*:5556") as client:
    client.listen(callback=handle_transcription)
```

See `example_consumer.py` for complete examples.

## Sending Audio (Upstream Services)

```python
import zmq
from src.stt.messaging.schemas import AudioRequest
from src.stt.messaging.serialization import serialize_audio_request

context = zmq.Context()
socket = context.socket(zmq.DEALER)
socket.connect("tcp://localhost:5555")

with open("audio.wav", "rb") as f:
    audio_data = f.read()

request = AudioRequest(
    request_id="123",
    audio_format="wav",
    sample_rate=16000,
    audio_data=audio_data,
)

socket.send(serialize_audio_request(request))
```

See `test_client.py` for a complete example.

```

## Notes

- Model loads on first request, not startup
- Model deallocates after timeout to free memory (~2GB GPU)
- All errors logged and returned as error responses
- Log rotation: 10MB max, weekly backup
