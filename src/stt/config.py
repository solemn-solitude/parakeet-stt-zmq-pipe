"""Configuration dataclasses for the STT service."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ainet.config import ConfigStore


_STORE = ConfigStore()


@dataclass
class STTConfig:
    """Main configuration for the STT service.

    Priority: env var > config.toml > hardcoded default.
    """

    # ZMQ addresses
    input_address: str = field(default_factory=lambda: os.getenv("STT_INPUT_ADDRESS", "tcp://localhost:20499"))
    output_address: str = field(default_factory=lambda: os.getenv("LLM_RAG_PIPE_INPUT_ADDRESS", "tcp://localhost:5555"))

    # Model configuration
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2"
    model_timeout_minutes: int = 10

    # Audio processing
    convert_to_mono: bool = False
    expected_sample_rate: int = 16000

    # Logging configuration
    log_file: Path = field(default_factory=lambda: Path("stt.log"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_days: int = 7

    def __post_init__(self):
        if isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)
        if not os.getenv("STT_INPUT_ADDRESS"):
            if v := _STORE.get("stt", "input_address"):
                self.input_address = v
        if not os.getenv("LLM_RAG_PIPE_INPUT_ADDRESS"):
            if v := _STORE.get("stt", "output_address"):
                self.output_address = v
        if not os.getenv("STT_MODEL_NAME"):
            if v := _STORE.get("stt", "model_name"):
                self.model_name = v
