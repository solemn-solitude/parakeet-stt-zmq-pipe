"""ZMQ socket handler for ROUTER (input) and DEALER (output) communication."""
import logging

import zmq

from stt.messaging.schemas import AudioRequest, TranscriptionResponse
from stt.messaging.serialization import deserialize_audio_request, serialize_transcription_response


logger = logging.getLogger(__name__)


class ZMQHandler:
    """Manages ZMQ ROUTER (input) and DEALER (output) sockets."""

    def __init__(self, input_address: str, output_address: str):
        self.input_address = input_address
        self.output_address = output_address

        self.context: zmq.Context | None = None
        self.input_socket: zmq.Socket | None = None
        self.output_socket: zmq.Socket | None = None

        logger.info(
            "ZMQHandler initialized with input=%s, output=%s",
            input_address,
            output_address,
        )

    def setup(self) -> None:
        logger.info("Setting up ZMQ sockets...")

        self.context = zmq.Context()

        self.input_socket = self.context.socket(zmq.ROUTER)
        self.input_socket.bind(self.input_address)
        logger.info("ROUTER socket bound to %s", self.input_address)

        self.output_socket = self.context.socket(zmq.DEALER)
        self.output_socket.connect(self.output_address)
        logger.info("DEALER socket connected to %s", self.output_address)

    def receive_request(self, timeout_ms: int = 100) -> tuple[bytes, AudioRequest] | None:
        """Receive an audio request from the input socket.

        Returns:
            Tuple of (client_identity, AudioRequest) if message received, None on timeout.

        Raises:
            ValueError: If message deserialization fails
        """
        if not self.input_socket:
            raise RuntimeError("Input socket not initialized. Call setup() first.")

        poller = zmq.Poller()
        poller.register(self.input_socket, zmq.POLLIN)

        events = dict(poller.poll(timeout_ms))

        if self.input_socket not in events:
            return None

        try:
            # ROUTER receives: [identity, empty, data]
            message_parts = self.input_socket.recv_multipart()

            if len(message_parts) < 2:
                logger.error(
                    "Invalid message format: expected at least 2 parts, got %d",
                    len(message_parts),
                )
                return None

            identity = message_parts[0]
            # message_parts[1] is the delimiter (empty frame)
            request_data = message_parts[-1] if len(message_parts) > 1 else message_parts[0]

            request = deserialize_audio_request(request_data)

            logger.debug(
                "Received request %s from client %s",
                request.request_id,
                identity.hex(),
            )
            return identity, request

        except ValueError as e:
            logger.error("Failed to deserialize request: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error receiving request: %s", e, exc_info=True)
            return None

    def send_response(self, response: TranscriptionResponse) -> None:
        if not self.output_socket:
            raise RuntimeError("Output socket not initialized. Call setup() first.")

        try:
            response_data = serialize_transcription_response(response)

            # DEALER sends topic + data so LLM ROUTER can route by MessageTopic.STT
            self.output_socket.send_multipart([b"stt", response_data])

            logger.debug(
                "Sent response for request %s (status=%s)",
                response.request_id,
                response.status,
            )
            logger.debug("Transcribed text: %s", response.text)

        except Exception as e:
            logger.error(
                "Failed to send response for request %s: %s",
                response.request_id,
                e,
                exc_info=True,
            )
            raise

    def cleanup(self) -> None:
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
