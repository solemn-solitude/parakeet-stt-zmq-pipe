"""Configuration dataclasses for the STT service."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class STTConfig:
    """Main configuration for the STT service."""
    
    # ZMQ addresses
    input_address: str = field(default_factory=lambda: os.getenv("STT_INPUT_ADDRESS", "tcp://localhost:20499"))
    output_address: str = field(default_factory=lambda: os.getenv("LLM_RAG_PIPE_INPUT_ADDRESS", "tcp://localhost:25000"))
    
    # Model configuration
    model_name: str = "nvidia/parakeet-tdt-0.6b-v2"
    model_timeout_minutes: int = 10
    
    # Audio processing
    convert_to_mono: bool = False
    expected_sample_rate: int = 16000
    
    # Logging configuration
    log_file: Path = field(default_factory=lambda: Path("stt.log"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_backup_days: int = 7
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)
