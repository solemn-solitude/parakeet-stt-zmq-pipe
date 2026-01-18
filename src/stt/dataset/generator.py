"""Dataset generation for TTS training."""
import logging
from pathlib import Path
from typing import Set
from textwrap import dedent

import click

from ..connection.sqlite_connection import SQLiteConnection
from ..core.model_manager import ModelManager
from ..core.transcription import TranscriptionEngine


logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate TTS training datasets from audio files."""
    
    def __init__(self, wav_directory: Path, voice_actor_identifier: str):
        """Initialize dataset generator.
        
        Args:
            wav_directory: Directory containing .wav files
            voice_actor_identifier: Identifier for the voice actor
        """
        self.wav_directory = wav_directory
        self.voice_actor_identifier = voice_actor_identifier
        self.metadata_file = wav_directory / f"{voice_actor_identifier}.metadata.txt"
        self.db_file = wav_directory / f"{voice_actor_identifier}.metadata.db"
        
        # Initialize transcription components
        self.model_manager = ModelManager()
        self.transcription_engine = TranscriptionEngine(self.model_manager)
    
    def _init_database(self):
        """Initialize the SQLite database with required table."""
        with SQLiteConnection(self.db_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    file_id TEXT PRIMARY KEY,
                    wav_file TEXT NOT NULL,
                    transcription TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info(f"Database initialized at {self.db_file}")
    
    def _get_transcribed_files(self) -> dict[str, tuple[str, str]]:
        """Get all transcribed files from the database.
        
        Returns:
            Dict mapping file_id to (wav_file, transcription)
        """
        with SQLiteConnection(self.db_file) as conn:
            conn.execute("SELECT file_id, wav_file, transcription FROM transcriptions")
            rows = conn.fetchall()
            return {row['file_id']: (row['wav_file'], row['transcription']) for row in rows}
    
    def _get_metadata_file_ids(self) -> Set[str]:
        """Get all file IDs from the metadata file.
        
        Returns:
            Set of file IDs present in metadata file
        """
        if not self.metadata_file.exists():
            return set()
        
        file_ids = set()
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '|' in line:
                    file_id = line.split('|', 1)[0]
                    file_ids.add(file_id)
        
        return file_ids
    
    def _cleanup_orphaned_entries(self):
        """Remove database entries not present in metadata file."""
        metadata_ids = self._get_metadata_file_ids()
        
        with SQLiteConnection(self.db_file) as conn:
            conn.execute("SELECT file_id FROM transcriptions")
            db_ids = {row['file_id'] for row in conn.fetchall()}
            
            orphaned_ids = db_ids - metadata_ids
            
            if orphaned_ids:
                logger.info(f"Removing {len(orphaned_ids)} orphaned database entries")
                placeholders = ','.join('?' * len(orphaned_ids))
                conn.execute(
                    f"DELETE FROM transcriptions WHERE file_id IN ({placeholders})",
                    tuple(orphaned_ids)
                )
    
    def _save_transcription(self, file_id: str, wav_file: str, transcription: str):
        """Save transcription to database and metadata file.
        
        Args:
            file_id: File identifier (filename without extension)
            wav_file: Path to the wav file
            transcription: Transcription text
        """
        # Save to database
        with SQLiteConnection(self.db_file) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO transcriptions (file_id, wav_file, transcription) VALUES (?, ?, ?)",
                (file_id, wav_file, transcription)
            )
        
        # Append to metadata file
        with open(self.metadata_file, 'a', encoding='utf-8') as f:
            f.write(f"{file_id}|{transcription}\n")
    
    def generate(self):
        """Generate the dataset by transcribing all .wav files."""
        # Initialize database
        self._init_database()
        
        # Get list of all .wav files
        wav_files = list(self.wav_directory.glob("*.wav"))
        
        if not wav_files:
            click.echo(f"No .wav files found in {self.wav_directory}")
            return
        
        click.echo(f"Found {len(wav_files)} .wav files in {self.wav_directory}")
        
        # Get already transcribed files
        transcribed = self._get_transcribed_files()
        
        # Create metadata file if it doesn't exist
        if not self.metadata_file.exists():
            self.metadata_file.touch()
        
        # Clean up orphaned database entries
        self._cleanup_orphaned_entries()
        
        # Process each wav file
        processed = 0
        skipped = 0
        
        with click.progressbar(
            wav_files,
            label="Transcribing audio files",
            show_pos=True
        ) as files:
            for wav_file in files:
                file_id = wav_file.stem  # Filename without extension
                
                # Skip if already transcribed
                if file_id in transcribed:
                    skipped += 1
                    logger.debug(f"Skipping already transcribed file: {file_id}")
                    continue
                
                try:
                    # Transcribe the audio file
                    transcription, processing_time_ms, _ = self.transcription_engine.transcribe(wav_file)
                    
                    # Save the transcription
                    self._save_transcription(file_id, str(wav_file), transcription)
                    
                    processed += 1
                    logger.info(
                        f"Transcribed {file_id}: {len(transcription)} chars "
                        f"in {processing_time_ms:.2f}ms"
                    )
                    
                except Exception as e:
                    click.echo(f"\nError transcribing {wav_file.name}: {e}", err=True)
                    logger.error(f"Failed to transcribe {wav_file}: {e}", exc_info=True)
        
        # Summary
        click.echo(dedent(f"""
            ============================================================
            Dataset Generation Complete
            ============================================================
              Total files:         {len(wav_files)}
              Newly transcribed:   {processed}
              Skipped (existing):  {skipped}
              Metadata file:       {self.metadata_file}
              Database file:       {self.db_file}
            ============================================================
            """))
