"""
CLI subcommands for creating tables, vocabulary, and data in a target synthetic database.

All commands here belong to the "Create Synthetic Database" rich help panel in the CLI.
"""

from pathlib import Path
from typing import Optional

from typer import Exit, Option

from datafaker.cli._defaults import CONFIG_FILENAME, ORM_FILENAME, STATS_FILENAME
from datafaker.cli._load_metadata import load_metadata_config, load_metadata_for_output
from datafaker.create import create_db_data, create_db_tables, create_db_vocab
from datafaker.db_utils import sorted_non_vocabulary_tables
from datafaker.serialize_metadata import dict_to_metadata
from datafaker.settings import SettingsError
from datafaker.utils import generators_require_stats, logger, read_config_file


def create_tables(
    orm_file: Path = Option(
        ORM_FILENAME,
        help="The name of the ORM yaml file",
        dir_okay=False,
    ),
    config_file: Optional[Path] = Option(
        CONFIG_FILENAME,
        help="The configuration file",
        dir_okay=False,
    ),
) -> None:
    """Create schema from the ORM YAML file.

    This CLI command creates the destination schema using object
    relational model declared as Python tables.

    Example:
        $ datafaker create-tables
    """
    logger.debug("Creating tables.")
    config = read_config_file(config_file) if config_file is not None else {}
    orm_metadata = load_metadata_for_output(orm_file, config)
    create_db_tables(orm_metadata)
    logger.debug("Tables created.")


def create_vocab(
    orm_file: Path = Option(
        ORM_FILENAME,
        help="The name of the ORM yaml file",
        dir_okay=False,
    ),
    config_file: Path = Option(
        CONFIG_FILENAME,
        help="The configuration file",
        dir_okay=False,
    ),
) -> None:
    """Import vocabulary data into the target database.

    Example:
        $ datafaker create-vocab
    """
    logger.debug("Loading vocab.")
    config = read_config_file(config_file) if config_file is not None else {}
    meta_dict = load_metadata_config(orm_file, config)
    orm_metadata = dict_to_metadata(meta_dict, config)
    vocabs_loaded = create_db_vocab(orm_metadata, meta_dict, config)
    num_vocabs = len(vocabs_loaded)
    logger.debug("%s %s loaded.", num_vocabs, "table" if num_vocabs == 1 else "tables")


def create_data(
    orm_file: Path = Option(
        ORM_FILENAME,
        help="The name of the ORM yaml file",
        dir_okay=False,
    ),
    config_file: Optional[Path] = Option(
        CONFIG_FILENAME,
        help="The configuration file",
    ),
    stats_file: Optional[Path] = Option(
        None,
        help=(
            "Statistics file (output of make-stats); default is src-stats.yaml if the "
            "config file references SRC_STATS, or None otherwise."
        ),
        show_default=False,
        dir_okay=False,
    ),
    num_passes: int = Option(1, help="Number of passes (rows or stories) to make"),
) -> None:
    """Populate the schema in the target directory with synthetic data.

    This CLI command generates synthetic data for
    Python table structures, and inserts these rows
    into a destination schema.

    Also takes as input object relational model as represented
    by file containing Python classes and its attributes.

    Takes as input datafaker output as represented by Python
    classes, its attributes and methods for generating values
    for those attributes.

    Final input is the number of rows required.

    Example:
        $ datafaker create-data
    """
    logger.debug("Creating data.")
    config = read_config_file(config_file) if config_file is not None else {}
    if stats_file is None and generators_require_stats(config):
        stats_file = Path(STATS_FILENAME)
    orm_metadata = load_metadata_for_output(orm_file, config)
    try:
        row_counts = create_db_data(
            sorted_non_vocabulary_tables(orm_metadata, config),
            config,
            stats_file,
            num_passes,
            orm_metadata,
        )
        logger.debug(
            "Data created in %s %s.",
            num_passes,
            "pass" if num_passes == 1 else "passes",
        )
        for table_name, row_count in row_counts.items():
            logger.debug(
                "%s: %s %s created.",
                table_name,
                row_count,
                "row" if row_count == 1 else "rows",
            )
        return
    except Exit:
        raise
    except RuntimeError as e:
        logger.error(e.args[0])
    except SettingsError as e:
        logger.error(str(e))
    raise Exit(1)
