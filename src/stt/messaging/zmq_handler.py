"""ZMQ socket handler for ROUTER (input) and DEALER (output) communication."""
import logging
import zmq
from typing import Optional, Tuple

from .schemas import AudioRequest, TranscriptionResponse
from .serialization import deserialize_audio_request, serialize_transcription_response


logger = logging.getLogger(__name__)


class ZMQHandler:
    """Manages ZMQ ROUTER (input) and DEALER (output) sockets."""
    
    def __init__(self, input_address: str, output_address: str):
        """Initialize ZMQ handler.
        
        Args:
            input_address: Address to bind ROUTER socket (e.g., tcp://*:5555)
            output_address: Address to connect DEALER socket (e.g., tcp://localhost:5556)
        """
        self.input_address = input_address
        self.output_address = output_address
        
        self.context: Optional[zmq.Context] = None
        self.input_socket: Optional[zmq.Socket] = None
        self.output_socket: Optional[zmq.Socket] = None
        
        logger.info(f"ZMQHandler initialized with input={input_address}, output={output_address}")
    
    def setup(self) -> None:
        """Set up ZMQ context and sockets."""
        logger.info("Setting up ZMQ sockets...")
        
        self.context = zmq.Context()
        
        # ROUTER socket for receiving requests (acts as server)
        self.input_socket = self.context.socket(zmq.ROUTER)
        self.input_socket.bind(self.input_address)
        logger.info(f"ROUTER socket bound to {self.input_address}")
        
        # DEALER socket for sending responses (acts as client)
        self.output_socket = self.context.socket(zmq.DEALER)
        self.output_socket.connect(self.output_address)
        logger.info(f"DEALER socket connected to {self.output_address}")
    
    def receive_request(self, timeout_ms: int = 100) -> Optional[Tuple[bytes, AudioRequest]]:
        """Receive an audio request from the input socket.
        
        Args:
            timeout_ms: Timeout in milliseconds for polling
            
        Returns:
            Tuple of (client_identity, AudioRequest) if message received, None otherwise
            
        Raises:
            ValueError: If message deserialization fails
        """
        if not self.input_socket:
            raise RuntimeError("Input socket not initialized. Call setup() first.")
        
        # Poll with timeout to avoid blocking indefinitely
        poller = zmq.Poller()
        poller.register(self.input_socket, zmq.POLLIN)
        
        events = dict(poller.poll(timeout_ms))
        
        if self.input_socket not in events:
            return None
        
        try:
            # ROUTER receives: [identity, empty, data]
            print("Waiting...")
            message_parts = self.input_socket.recv_multipart()
            #print(f"Message parts: {message_parts}")
            print("received...")
            
            if len(message_parts) < 2:
                print("Err, wrong message format length, expected two parts")
                logger.error(f"Invalid message format: expected at least 2 parts, got {len(message_parts)}")
                return None
            
            identity = message_parts[0]
            print(f"identitiy: {identity}")
            # message_parts[1] is the delimiter (empty frame)
            request_data = message_parts[-1] if len(message_parts) > 1 else message_parts[0]
            print("request data fetched")
            
            # Deserialize the request
            request = deserialize_audio_request(request_data)
            print("deserialized")
            
            logger.debug(f"Received request {request.request_id} from client {identity.hex()}")
            return identity, request
            
        except ValueError as e:
            logger.error(f"Failed to deserialize request: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error receiving request: {e}", exc_info=True)
            return None
    
    def send_response(self, response: TranscriptionResponse) -> None:
        """Send a transcription response to the output socket.
        
        Args:
            response: The TranscriptionResponse to send
        """
        if not self.output_socket:
            raise RuntimeError("Output socket not initialized. Call setup() first.")
        
        try:
            # Serialize the response
            response_data = serialize_transcription_response(response)
            
            # DEALER sends without identity frame
            self.output_socket.send(response_data)
            
            logger.debug(f"Sent response for request {response.request_id} (status={response.status})")
            logger.debug(f"Transcribed text: {response.text}")
            
        except Exception as e:
            logger.error(f"Failed to send response for request {response.request_id}: {e}", exc_info=True)
            raise
    
    def cleanup(self) -> None:
        """Clean up ZMQ sockets and context."""
        logger.info("Cleaning up ZMQ resources...")
        
        if self.input_socket:
            self.input_socket.close()
            self.input_socket = None
            logger.debug("Input socket closed")
        
        if self.output_socket:
            self.output_socket.close()
            self.output_socket = None
            logger.debug("Output socket closed")
        
        if self.context:
            self.context.term()
            self.context = None
            logger.debug("ZMQ context terminated")
        
        logger.info("ZMQ cleanup complete")
