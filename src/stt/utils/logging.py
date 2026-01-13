"""Logging configuration with file rotation capabilities."""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import time
import threading


class TimedAndSizeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """File handler that rotates based on both size and time."""
    
    def __init__(
        self,
        filename: Path,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_days: int = 7,
        encoding: Optional[str] = "utf-8",
    ):
        """Initialize the rotating file handler.
        
        Args:
            filename: Path to the log file
            max_bytes: Maximum size in bytes before rotation
            backup_days: Number of days to keep backup files
            encoding: File encoding
        """
        super().__init__(
            filename=str(filename),
            maxBytes=max_bytes,
            backupCount=backup_days,
            encoding=encoding,
        )
        self.last_check_time = time.time()
        self.check_interval = 3600  # Check every hour
        
    def shouldRollover(self, record: logging.LogRecord) -> int:
        """Determine if rollover should occur.
        
        Checks both file size and time since last check.
        
        Args:
            record: The log record
            
        Returns:
            1 if rollover should occur, 0 otherwise
        """
        # Check size-based rollover (from parent class)
        if super().shouldRollover(record):
            return 1
        
        # Check time-based rollover
        current_time = time.time()
        if current_time - self.last_check_time >= self.check_interval:
            self.last_check_time = current_time
            # Force a flush every hour
            self.flush()
        
        return 0


def setup_logging(
    log_file: Path,
    log_level: str = "WARNING",
    log_max_bytes: int = 10 * 1024 * 1024,
    log_backup_days: int = 7,
) -> None:
    """Set up logging configuration for the application.
    
    Args:
        log_file: Path to the log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_max_bytes: Maximum log file size before rotation
        log_backup_days: Number of days to keep backup files
    """
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create rotating file handler
    file_handler = TimedAndSizeRotatingFileHandler(
        filename=log_file,
        max_bytes=log_max_bytes,
        backup_days=log_backup_days,
    )
    file_handler.setFormatter(formatter)
    
    # Create console handler for errors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.ERROR)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove any existing handlers
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger("nemo").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized: file={log_file}, level={log_level}")


class PeriodicFlusher:
    """Background thread that periodically flushes log handlers."""
    
    def __init__(self, interval_seconds: int = 3600):
        """Initialize the periodic flusher.
        
        Args:
            interval_seconds: How often to flush (default: 1 hour)
        """
        self.interval = interval_seconds
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start the flusher thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.thread.start()
        logging.debug("Periodic log flusher started")
    
    def stop(self) -> None:
        """Stop the flusher thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logging.debug("Periodic log flusher stopped")
    
    def _flush_loop(self) -> None:
        """Background loop that flushes handlers periodically."""
        while self.running:
            time.sleep(self.interval)
            if self.running:
                self._flush_all_handlers()
    
    def _flush_all_handlers(self) -> None:
        """Flush all handlers in the root logger."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            try:
                handler.flush()
            except Exception as e:
                logging.error(f"Failed to flush handler {handler}: {e}")
