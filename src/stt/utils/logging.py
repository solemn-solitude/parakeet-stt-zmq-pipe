"""Logging configuration with file rotation capabilities."""
import logging
import logging.handlers
import threading
import time
from pathlib import Path


class TimedAndSizeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """File handler that rotates based on both size and time."""

    def __init__(
        self,
        filename: Path,
        max_bytes: int = 10 * 1024 * 1024,
        backup_days: int = 7,
        encoding: str | None = "utf-8",
    ):
        super().__init__(
            filename=str(filename),
            maxBytes=max_bytes,
            backupCount=backup_days,
            encoding=encoding,
        )
        self.last_check_time = time.time()
        self.check_interval = 3600

    def shouldRollover(self, record: logging.LogRecord) -> int:
        if super().shouldRollover(record):
            return 1

        current_time = time.time()
        if current_time - self.last_check_time >= self.check_interval:
            self.last_check_time = current_time
            self.flush()

        return 0


def setup_logging(
    log_file: Path,
    log_level: str = "WARNING",
    log_max_bytes: int = 10 * 1024 * 1024,
    log_backup_days: int = 7,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = TimedAndSizeRotatingFileHandler(
        filename=log_file,
        max_bytes=log_max_bytes,
        backup_days=log_backup_days,
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.ERROR)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("nemo").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    logging.info("Logging initialized: file=%s, level=%s", log_file, log_level)


class PeriodicFlusher:
    """Background thread that periodically flushes log handlers."""

    def __init__(self, interval_seconds: int = 3600):
        self.interval = interval_seconds
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._flush_loop, daemon=True)
        self.thread.start()
        logging.debug("Periodic log flusher started")

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logging.debug("Periodic log flusher stopped")

    def _flush_loop(self) -> None:
        while self.running:
            time.sleep(self.interval)
            if self.running:
                self._flush_all_handlers()

    def _flush_all_handlers(self) -> None:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            try:
                handler.flush()
            except Exception as e:
                logging.error("Failed to flush handler %s: %s", handler, e)
