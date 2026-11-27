"""
CLI subcommands for removing tables, vocabulary, and data in a target synthetic database.

All commands here belong to the "Remove Destination Data" rich help panel in the CLI.
"""

from pathlib import Path
from typing import Optional

import typer
from sqlalchemy.exc import InternalError

from datafaker.cli._defaults import CONFIG_FILENAME, ORM_FILENAME
from datafaker.cli._load_metadata import load_metadata_config, load_metadata_for_output
from datafaker.remove import remove_db_data, remove_db_tables, remove_db_vocab
from datafaker.serialize_metadata import dict_to_metadata
from datafaker.utils import logger, read_config_file


def remove_data(
    orm_file: Path = typer.Option(
        ORM_FILENAME,
        help="The name of the ORM yaml file",
        dir_okay=False,
    ),
    config_file: Optional[Path] = typer.Option(
        CONFIG_FILENAME,
        help="The configuration file",
        dir_okay=False,
    ),
    yes: bool = typer.Option(
        False, "--yes", prompt="Are you sure?", help="Just remove, don't ask first"
    ),
) -> None:
    """Truncate non-vocabulary tables in the destination schema."""
    if yes:
        logger.debug("Truncating non-vocabulary tables.")
        config = read_config_file(config_file) if config_file is not None else {}
        metadata = load_metadata_for_output(orm_file, config)
        remove_db_data(metadata, config)
        logger.debug("Non-vocabulary tables truncated.")
    else:
        logger.info("Would truncate non-vocabulary tables if called with --yes.")


def remove_vocab(
    orm_file: Path = typer.Option(
        ORM_FILENAME,
        help="The name of the ORM yaml file",
        dir_okay=False,
    ),
    config_file: Optional[Path] = typer.Option(
        CONFIG_FILENAME,
        help="The configuration file",
        dir_okay=False,
    ),
    yes: bool = typer.Option(
        False, "--yes", prompt="Are you sure?", help="Just remove, don't ask first"
    ),
) -> None:
    """Truncate vocabulary tables in the destination schema."""
    if yes:
        logger.debug("Truncating vocabulary tables.")
        config = read_config_file(config_file) if config_file is not None else {}
        meta_dict = load_metadata_config(orm_file, config)
        orm_metadata = dict_to_metadata(meta_dict, config)
        remove_db_vocab(orm_metadata, meta_dict, config)
        logger.debug("Vocabulary tables truncated.")
    else:
        logger.info("Would truncate vocabulary tables if called with --yes.")


def remove_tables(
    orm_file: Path = typer.Option(
        ORM_FILENAME,
        help="The name of the ORM yaml file",
        dir_okay=False,
    ),
    config_file: Path = typer.Option(
        CONFIG_FILENAME,
        help="The configuration file",
        dir_okay=False,
    ),
    # pylint: disable=redefined-builtin
    all: bool = typer.Option(
        False,
        help="Don't use the ORM file, delete all tables in the destination schema",
    ),
    yes: bool = typer.Option(
        False, "--yes", prompt="Are you sure?", help="Just remove, don't ask first"
    ),
) -> None:
    """Drop all tables in the destination schema.

    Does not drop the schema itself.
    """
    if yes:
        logger.debug("Dropping tables.")
        if all:
            remove_db_tables(None)
        else:
            config = read_config_file(config_file)
            metadata = load_metadata_for_output(orm_file, config)
            try:
                remove_db_tables(metadata)
            except InternalError as exc:
                logger.error("Failed to drop tables: %s", exc)
                logger.error("Please try again using the --all option.")
                raise typer.Exit(1) from exc
        logger.debug("Tables dropped.")
    else:
        logger.info("Would remove tables if called with --yes.")
