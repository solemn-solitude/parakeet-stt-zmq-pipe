"""Command-line interface for the STT service."""
import click
from pathlib import Path

from .config import STTConfig
from .service import STTService
from .utils.logging import setup_logging


@click.group()
def cli():
    """Speech-to-Text (STT) service using NeMo Parakeet model."""
    pass


@cli.command()
@click.option(
    "--input-address",
    default="tcp://*:5555",
    help="ZMQ ROUTER input address (default: tcp://*:5555)",
    show_default=True,
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
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
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
    
    # Create configuration
    config = STTConfig(
        input_address=input_address,
        output_address=output_address,
        model_timeout_minutes=timeout,
        convert_to_mono=convert_to_mono,
        log_file=Path(log_file),
        log_level=log_level.upper(),
    )
    
    # Set up logging
    setup_logging(
        log_file=config.log_file,
        log_level=config.log_level,
        log_max_bytes=config.log_max_bytes,
        log_backup_days=config.log_backup_days,
    )
    
    # Print startup information
    click.echo("=" * 60)
    click.echo("STT Service Configuration:")
    click.echo("=" * 60)
    click.echo(f"  Input Address:      {config.input_address}")
    click.echo(f"  Output Address:     {config.output_address}")
    click.echo(f"  Model:              {config.model_name}")
    click.echo(f"  Model Timeout:      {config.model_timeout_minutes} minutes")
    click.echo(f"  Convert to Mono:    {config.convert_to_mono}")
    click.echo(f"  Log File:           {config.log_file}")
    click.echo(f"  Log Level:          {config.log_level}")
    click.echo("=" * 60)
    click.echo()
    
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


if __name__ == "__main__":
    cli()
