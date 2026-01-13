"""Simple test client for the STT service."""
import zmq
import time
from pathlib import Path

from src.stt.messaging.schemas import AudioRequest, TranscriptionResponse
from src.stt.messaging.serialization import (
    serialize_audio_request,
    deserialize_transcription_response,
)


def test_stt_service():
    """Test the STT service with a sample audio file."""
    
    # Create ZMQ context and DEALER socket (client)
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect("tcp://localhost:5555")
    
    print("Connected to STT service at tcp://localhost:5555")
    
    try:
        # Load the test audio file
        audio_file = Path("sora_sample.wav")
        if not audio_file.exists():
            print(f"Error: Audio file not found: {audio_file}")
            return
        
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        
        print(f"Loaded audio file: {audio_file} ({len(audio_data)} bytes)")
        
        # Create request
        request = AudioRequest(
            request_id="test-001",
            audio_format="wav",
            sample_rate=16000,
            audio_data=audio_data,
        )
        
        print(f"Sending request: {request.request_id}")
        
        # Serialize and send
        request_bytes = serialize_audio_request(request)
        socket.send(request_bytes)
        
        print("Waiting for response...")
        
        # Wait for response (with timeout)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        
        if poller.poll(30000):  # 30 second timeout
            response_bytes = socket.recv()
            response = deserialize_transcription_response(response_bytes)
            
            print("\n" + "=" * 60)
            print("RESPONSE RECEIVED")
            print("=" * 60)
            print(f"Request ID:      {response.request_id}")
            print(f"Status:          {response.status}")
            print(f"Processing Time: {response.processing_time_ms:.2f}ms")
            
            if response.status == "success":
                print(f"Confidence:      {response.confidence}")
                print(f"\nTranscription:\n{response.text}")
            else:
                print(f"Error:           {response.error_details}")
            
            print("=" * 60)
        else:
            print("ERROR: Timeout waiting for response")
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        socket.close()
        context.term()
        print("\nConnection closed")


if __name__ == "__main__":
    test_stt_service()
