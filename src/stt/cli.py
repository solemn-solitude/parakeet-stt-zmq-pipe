"""Command-line interface for the STT service."""

import click
from pathlib import Path
from textwrap import dedent

from .config import STTConfig
from .service import STTService
from .utils.logging import setup_logging
from .dataset.generator import DatasetGenerator

@click.group()
def cli():
    """Speech-to-Text (STT) service using NeMo Parakeet model."""
    pass


@cli.command()
@click.option(
    "--input-address",
    default=None,
    help="ZMQ ROUTER input address (overrides STT_INPUT_ADDRESS env var, default: tcp://localhost:20499)",
)
@click.option(
    "--output-address",
    default="tcp://localhost:5556",
    help="ZMQ DEALER output address (default: tcp://localhost:5556)",
    show_default=True,
)
@click.option(
    "--timeout",
    default=10,
    type=int,
    help="Model idle timeout in minutes before deallocation (default: 10)",
    show_default=True,
)
@click.option(
    "--convert-to-mono",
    is_flag=True,
    default=False,
    help="Enable automatic stereo to mono conversion (default: disabled)",
)
@click.option(
    "--log-file",
    default="stt.log",
    type=click.Path(),
    help="Log file path (default: stt.log)",
    show_default=True,
)
@click.option(
    "--log-level",
    default="WARNING",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Logging level (default: WARNING)",
    show_default=True,
)
def start(
    input_address: str,
    output_address: str,
    timeout: int,
    convert_to_mono: bool,
    log_file: str,
    log_level: str,
):
    """Start the STT service."""

    # Create configuration with conditional input_address override
    config_kwargs = {
        "output_address": output_address,
        "model_timeout_minutes": timeout,
        "convert_to_mono": convert_to_mono,
        "log_file": Path(log_file),
        "log_level": log_level.upper(),
    }
    
    # Only override input_address if explicitly provided via CLI
    if input_address is not None:
        config_kwargs["input_address"] = input_address
    
    config = STTConfig(**config_kwargs)

    # Set up logging
    setup_logging(
        log_file=config.log_file,
        log_level=config.log_level,
        log_max_bytes=config.log_max_bytes,
        log_backup_days=config.log_backup_days,
    )

    # Print startup information
    _print_stt_service_configuration(config)

    # Create and run service
    service = STTService(config)

    try:
        service.run()
    except KeyboardInterrupt:
        click.echo("\nShutdown requested by user")
    except Exception as e:
        click.echo(f"\nFatal error: {e}", err=True)
        raise


@cli.command()
def version():
    """Show version information."""
    click.echo("Parakeet STT ZMQ Pipe v0.1.0")
    click.echo("Using NeMo Parakeet TDT 0.6B v2 model")


@cli.command(name="generate-soprano-dataset")
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("voice_actor_identifier")
@click.option(
    "--log-file",
    default="dataset_generation.log",
    type=click.Path(),
    help="Log file path (default: dataset_generation.log)",
    show_default=True,
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    help="Logging level (default: INFO)",
    show_default=True,
)
def generate_soprano_dataset(
    directory: Path,
    voice_actor_identifier: str,
    log_file: str,
    log_level: str,
):
    """Generate TTS training dataset from .wav files.

    This command transcribes all .wav files in DIRECTORY and creates a metadata
    file in the format required for Soprano TTS training. It also maintains an
    SQLite database to track transcription progress and avoid re-processing files.

    DIRECTORY: Path to directory containing .wav files

    VOICE_ACTOR_IDENTIFIER: Identifier for the voice actor (used in output filenames)

    Output files:
    - {VOICE_ACTOR_IDENTIFIER}.metadata.txt: Metadata file in format "file_id|transcription"
    - {VOICE_ACTOR_IDENTIFIER}.metadata.db: SQLite database tracking transcription state
    """

    # Set up logging
    setup_logging(
        log_file=Path(log_file),
        log_level=log_level.upper(),
    )

    # Print startup information
    _print_soprano_dataset_generation_detail(
        directory, voice_actor_identifier, log_file, log_level
    )

    try:
        # Create generator and run
        generator = DatasetGenerator(directory, voice_actor_identifier)
        generator.generate()
    except KeyboardInterrupt:
        click.echo("\nDataset generation interrupted by user")
    except Exception as e:
        click.echo(f"\nFatal error: {e}", err=True)
        raise


def _print_soprano_dataset_generation_detail(
    directory, voice_actor_identifier, log_file, log_level
):
    click.echo(
        dedent(f"""
        ============================================================
        Soprano Dataset Generation
        ============================================================
          Directory:              {directory}
          Voice Actor ID:         {voice_actor_identifier}
          Log File:               {log_file}
          Log Level:              {log_level}
        ============================================================
        """)
    )


def _print_stt_service_configuration(config):
    click.echo(
        dedent(f"""
        ============================================================
        STT Service Configuration:
        ============================================================
          Input Address:      {config.input_address}
          Output Address:     {config.output_address}
          Model:              {config.model_name}
          Model Timeout:      {config.model_timeout_minutes} minutes
          Convert to Mono:    {config.convert_to_mono}
          Log File:           {config.log_file}
          Log Level:          {config.log_level}
        ============================================================
        """)
    )
