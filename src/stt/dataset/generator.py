"""Dataset generation for TTS training."""
import logging
from pathlib import Path
from textwrap import dedent

import click

from stt.connection.sqlite_connection import SQLiteConnection
from stt.core.model_manager import ModelManager
from stt.core.transcription import TranscriptionEngine


logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate TTS training datasets from audio files."""

    def __init__(self, wav_directory: Path, voice_actor_identifier: str):
        self.wav_directory = wav_directory
        self.voice_actor_identifier = voice_actor_identifier
        self.metadata_file = wav_directory / f"{voice_actor_identifier}.metadata.txt"
        self.db_file = wav_directory / f"{voice_actor_identifier}.metadata.db"

        self.model_manager = ModelManager()
        self.transcription_engine = TranscriptionEngine(self.model_manager)

    def _init_database(self) -> None:
        with SQLiteConnection(self.db_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    file_id TEXT PRIMARY KEY,
                    wav_file TEXT NOT NULL,
                    transcription TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("Database initialized at %s", self.db_file)

    def _get_transcribed_files(self) -> dict[str, tuple[str, str]]:
        with SQLiteConnection(self.db_file) as conn:
            conn.execute("SELECT file_id, wav_file, transcription FROM transcriptions")
            rows = conn.fetchall()
            return {row["file_id"]: (row["wav_file"], row["transcription"]) for row in rows}

    def _get_metadata_file_ids(self) -> set[str]:
        if not self.metadata_file.exists():
            return set()

        file_ids = set()
        with open(self.metadata_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "|" in line:
                    file_id = line.split("|", 1)[0]
                    file_ids.add(file_id)

        return file_ids

    def _cleanup_orphaned_entries(self) -> None:
        metadata_ids = self._get_metadata_file_ids()

        with SQLiteConnection(self.db_file) as conn:
            conn.execute("SELECT file_id FROM transcriptions")
            db_ids = {row["file_id"] for row in conn.fetchall()}

            orphaned_ids = db_ids - metadata_ids

            if orphaned_ids:
                logger.info("Removing %d orphaned database entries", len(orphaned_ids))
                placeholders = ",".join("?" * len(orphaned_ids))
                conn.execute(
                    f"DELETE FROM transcriptions WHERE file_id IN ({placeholders})",
                    tuple(orphaned_ids),
                )

    def _save_transcription(self, file_id: str, wav_file: str, transcription: str) -> None:
        with SQLiteConnection(self.db_file) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO transcriptions (file_id, wav_file, transcription) VALUES (?, ?, ?)",
                (file_id, wav_file, transcription),
            )

        with open(self.metadata_file, "a", encoding="utf-8") as f:
            f.write(f"{file_id}|{transcription}\n")

    def generate(self) -> None:
        self._init_database()

        wav_files = list(self.wav_directory.glob("*.wav"))

        if not wav_files:
            click.echo(f"No .wav files found in {self.wav_directory}")
            return

        click.echo(f"Found {len(wav_files)} .wav files in {self.wav_directory}")

        transcribed = self._get_transcribed_files()

        if not self.metadata_file.exists():
            self.metadata_file.touch()

        self._cleanup_orphaned_entries()

        processed = 0
        skipped = 0

        with click.progressbar(
            wav_files,
            label="Transcribing audio files",
            show_pos=True,
        ) as files:
            for wav_file in files:
                file_id = wav_file.stem

                if file_id in transcribed:
                    skipped += 1
                    logger.debug("Skipping already transcribed file: %s", file_id)
                    continue

                try:
                    transcription, processing_time_ms, _ = self.transcription_engine.transcribe(wav_file)

                    self._save_transcription(file_id, str(wav_file), transcription)

                    processed += 1
                    logger.info(
                        "Transcribed %s: %d chars in %.2fms",
                        file_id,
                        len(transcription),
                        processing_time_ms,
                    )

                except Exception as e:
                    click.echo(f"\nError transcribing {wav_file.name}: {e}", err=True)
                    logger.error("Failed to transcribe %s: %s", wav_file, e, exc_info=True)

        click.echo(
            dedent(f"""
            ============================================================
            Dataset Generation Complete
            ============================================================
              Total files:         {len(wav_files)}
              Newly transcribed:   {processed}
              Skipped (existing):  {skipped}
              Metadata file:       {self.metadata_file}
              Database file:       {self.db_file}
            ============================================================
            """)
        )
