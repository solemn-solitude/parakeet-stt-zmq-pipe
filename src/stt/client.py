"""Client library for consuming STT transcriptions in downstream services."""
import logging
from collections.abc import Callable

import zmq

from src.stt.messaging.schemas import TranscriptionResponse
from src.stt.messaging.serialization import deserialize_transcription_response


logger = logging.getLogger(__name__)


class STTClient:
    """Client for receiving STT transcription results.

    Designed for downstream services (like LLM/RAG) to consume transcriptions
    from the STT pipe.
    """

    def __init__(self, bind_address: str = "tcp://*:5556"):
        self.bind_address = bind_address
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self._running = False

        logger.info("STTClient initialized: bind_address=%s", bind_address)

    def connect(self) -> None:
        if self.socket is not None:
            logger.warning("Already connected")
            return

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(self.bind_address)

        logger.info("ROUTER socket bound to %s", self.bind_address)

    def receive(self, timeout_ms: int = 1000) -> TranscriptionResponse | None:
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
                logger.error("Invalid message format: %d parts", len(message_parts))
                return None

            response_data = message_parts[-1]
            response = deserialize_transcription_response(response_data)

            logger.debug(
                "Received transcription: request_id=%s, status=%s",
                response.request_id,
                response.status,
            )

            return response

        except ValueError as e:
            logger.error("Failed to deserialize response: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error receiving response: %s", e, exc_info=True)
            return None

    def listen(
        self,
        callback: Callable[[TranscriptionResponse], None],
        timeout_ms: int = 1000,
    ) -> None:
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
                            "Callback error for request %s: %s",
                            response.request_id,
                            e,
                            exc_info=True,
                        )
        except KeyboardInterrupt:
            logger.info("Listening interrupted by user")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        logger.info("Stopping listener")

    def disconnect(self) -> None:
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

    def __enter__(self) -> "STTClient":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.disconnect()
