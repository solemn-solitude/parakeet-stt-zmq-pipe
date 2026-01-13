"""Example of using STTClient in a downstream service (e.g., LLM/RAG)."""
import logging
from src.stt.client import STTClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def process_transcription(response):
    """Process a transcription response.
    
    This is where your LLM/RAG service would handle the transcribed text.
    """
    if response.status == "success":
        print(f"\n{'='*60}")
        print(f"Request ID: {response.request_id}")
        print(f"Transcription: {response.text}")
        print(f"Confidence: {response.confidence}")
        print(f"Processing Time: {response.processing_time_ms:.2f}ms")
        print(f"{'='*60}\n")
        
        # TODO: Send to LLM, do RAG lookup, etc.
        # llm_response = process_with_llm(response.text)
        # rag_results = query_knowledge_base(response.text)
        
    else:
        print(f"\nError for request {response.request_id}: {response.error_details}\n")


def main():
    """Example 1: Using context manager with callback."""
    print("Starting STT consumer (listening for transcriptions)...")
    print("Press Ctrl+C to stop\n")
    
    with STTClient(bind_address="tcp://*:5556") as client:
        # This will block and call process_transcription for each response
        client.listen(callback=process_transcription)


def main_manual():
    """Example 2: Manual connection with receive loop."""
    client = STTClient(bind_address="tcp://*:5556")
    client.connect()
    
    try:
        while True:
            # Poll for responses with 1 second timeout
            response = client.receive(timeout_ms=1000)
            
            if response:
                process_transcription(response)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.disconnect()


if __name__ == "__main__":
    # Use the callback-based approach (simpler)
    main()
    
    # Or use manual polling if you need more control
    # main_manual()
