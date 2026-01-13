"""Client library for consuming STT transcriptions in downstream services."""
import logging
import zmq
from typing import Optional, Callable

from .messaging.schemas import TranscriptionResponse
from .messaging.serialization import deserialize_transcription_response


logger = logging.getLogger(__name__)


class STTClient:
    """Client for receiving STT transcription results.
    
    This is designed for downstream services (like LLM/RAG) to consume
    transcriptions from the STT pipe.
    """
    
    def __init__(self, bind_address: str = "tcp://*:5556"):
        """Initialize the STT client.
        
        Args:
            bind_address: Address to bind ROUTER socket (where STT DEALER connects)
        """
        self.bind_address = bind_address
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self._running = False
        
        logger.info(f"STTClient initialized: bind_address={bind_address}")
    
    def connect(self) -> None:
        """Connect and set up the ROUTER socket."""
        if self.socket is not None:
            logger.warning("Already connected")
            return
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(self.bind_address)
        
        logger.info(f"ROUTER socket bound to {self.bind_address}")
    
    def receive(self, timeout_ms: int = 1000) -> Optional[TranscriptionResponse]:
        """Receive a single transcription response.
        
        Args:
            timeout_ms: Timeout in milliseconds (default: 1000ms)
            
        Returns:
            TranscriptionResponse if available, None if timeout
        """
        if not self.socket:
            raise RuntimeError("Not connected. Call connect() first.")
        
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        
        events = dict(poller.poll(timeout_ms))
        
        if self.socket not in events:
            return None
        
        try:
            # ROUTER receives: [identity, empty, data]
            message_parts = self.socket.recv_multipart()
            
            if len(message_parts) < 2:
                logger.error(f"Invalid message format: {len(message_parts)} parts")
                return None
            
            # Extract the actual data (last part)
            response_data = message_parts[-1]
            
            # Deserialize
            response = deserialize_transcription_response(response_data)
            
            logger.debug(
                f"Received transcription: request_id={response.request_id}, "
                f"status={response.status}"
            )
            
            return response
            
        except ValueError as e:
            logger.error(f"Failed to deserialize response: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error receiving response: {e}", exc_info=True)
            return None
    
    def listen(
        self,
        callback: Callable[[TranscriptionResponse], None],
        timeout_ms: int = 1000
    ) -> None:
        """Listen for transcriptions and invoke callback for each.
        
        Args:
            callback: Function to call with each TranscriptionResponse
            timeout_ms: Poll timeout in milliseconds
        """
        if not self.socket:
            raise RuntimeError("Not connected. Call connect() first.")
        
        self._running = True
        logger.info("Started listening for transcriptions")
        
        try:
            while self._running:
                response = self.receive(timeout_ms=timeout_ms)
                
                if response is not None:
                    try:
                        callback(response)
                    except Exception as e:
                        logger.error(
                            f"Callback error for request {response.request_id}: {e}",
                            exc_info=True
                        )
        except KeyboardInterrupt:
            logger.info("Listening interrupted by user")
        finally:
            self._running = False
    
    def stop(self) -> None:
        """Stop listening loop."""
        self._running = False
        logger.info("Stopping listener")
    
    def disconnect(self) -> None:
        """Disconnect and clean up resources."""
        logger.info("Disconnecting STT client")
        
        self._running = False
        
        if self.socket:
            self.socket.close()
            self.socket = None
            logger.debug("Socket closed")
        
        if self.context:
            self.context.term()
            self.context = None
            logger.debug("Context terminated")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
