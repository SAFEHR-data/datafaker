"""Entrypoint for the datafaker package."""
import asyncio
import importlib
import io
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Final, Optional

import yaml
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validate
from sqlalchemy import MetaData
from typer import Argument, Exit, Option, Typer

from datafaker.create import create_db_data, create_db_tables, create_db_vocab
from datafaker.dump import dump_db_tables
from datafaker.interactive import (
    update_config_generators,
    update_config_tables,
    update_missingness,
)
from datafaker.make import (
    make_src_stats,
    make_table_generators,
    make_tables_file,
    make_vocabulary_tables,
)
from datafaker.remove import remove_db_data, remove_db_tables, remove_db_vocab
from datafaker.settings import Settings, get_settings
from datafaker.utils import (
    CONFIG_SCHEMA_PATH,
    conf_logger,
    generators_require_stats,
    get_flag,
    import_file,
    logger,
    read_config_file,
    sorted_non_vocabulary_tables,
)

from .serialize_metadata import dict_to_metadata

# pylint: disable=too-many-arguments

ORM_FILENAME: Final[str] = "orm.yaml"
CONFIG_FILENAME: Final[str] = "config.yaml"
DF_FILENAME: Final[str] = "df.py"
STATS_FILENAME: Final[str] = "src-stats.yaml"

app = Typer(no_args_is_help=True)


def _check_file_non_existence(file_path: Path) -> None:
    """Check that a given file does not exist. Exit with an error message if it does."""
    if file_path.exists():
        logger.error("%s should not already exist. Exiting...", file_path)
        sys.exit(1)


def _require_src_db_dsn(settings: Settings) -> str:
    """Return the source DB DSN.

    Check that source DB details have been set. Exit with error message if not.
    """
    if (src_dsn := settings.src_dsn) is None:
        logger.error("Missing source database connection details.")
        sys.exit(1)
    return src_dsn


def load_metadata_config(
    orm_file_name: str, config: dict | None = None
) -> dict[str, Any]:
    """
    Load the ``orm.yaml`` file, returning a dict representation.

    :param orm_file_name: The name of the file to load.
    :param config: The ``config.yaml`` file object. Ignored tables will be
    excluded from the output.
    :return: A dict representing the ``orm.yaml`` file, with the tables
    the ``config`` says to ignore removed.
    """
    with open(orm_file_name, encoding="utf-8") as orm_fh:
        meta_dict = yaml.load(orm_fh, yaml.Loader)
        if not isinstance(meta_dict, dict):
            return {}
        tables_dict = meta_dict.get("tables", {})
        if config is not None and "tables" in config:
            # Remove ignored tables
            for name, table_config in config.get("tables", {}).items():
                if get_flag(table_config, "ignore"):
                    tables_dict.pop(name, None)
        return meta_dict


def load_metadata(orm_file_name: str, config: dict | None = None) -> MetaData:
    """
    Load metadata from ``orm.yaml``.

    :param orm_file_name: ``orm.yaml`` or alternative name to load metadata from.
    :param config: Used to exclude tables that are marked as ``ignore: true``.
    :return: SQLAlchemy MetaData object representing the database described by the loaded file.
    """
    meta_dict = load_metadata_config(orm_file_name, config)
    return dict_to_metadata(meta_dict, None)


def load_metadata_for_output(orm_file_name: str, config: dict | None = None) -> Any:
    """Load metadata excluding any foreign keys pointing to ignored tables."""
    meta_dict = load_metadata_config(orm_file_name, config)
    return dict_to_metadata(meta_dict, config)


@app.callback()
def main(
    verbose: bool = Option(False, "--verbose", "-v", help="Print more information.")
) -> None:
    """Set the global parameters."""
    conf_logger(verbose)


@app.command()
def create_data(
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
    df_file: str = Option(
        DF_FILENAME,
        help="The name of the generators file. Must be in the current working directory.",
    ),
    config_file: Optional[str] = Option(CONFIG_FILENAME, help="The configuration file"),
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
    orm_metadata = load_metadata_for_output(orm_file, config)
    df_module = import_file(df_file)
    try:
        row_counts = create_db_data(
            sorted_non_vocabulary_tables(orm_metadata, config),
            df_module,
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
    except RuntimeError as e:
        logger.error(e.args[0])
    raise Exit(1)


@app.command()
def create_vocab(
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
    config_file: str = Option(CONFIG_FILENAME, help="The configuration file"),
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


@app.command()
def create_tables(
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
    config_file: Optional[str] = Option(CONFIG_FILENAME, help="The configuration file"),
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


@app.command()
def create_generators(
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
    df_file: str = Option(DF_FILENAME, help="Path to write Python generators to."),
    config_file: str = Option(CONFIG_FILENAME, help="The configuration file"),
    stats_file: Optional[str] = Option(
        None,
        help=(
            "Statistics file (output of make-stats); default is src-stats.yaml if the "
            "config file references SRC_STATS, or None otherwise."
        ),
        show_default=False,
    ),
    force: bool = Option(
        False, "--force", "-f", help="Overwrite any existing Python generators file."
    ),
) -> None:
    """Make a datafaker file of generator classes.

    This CLI command takes an object relation model output by sqlcodegen and
    returns a set of synthetic data generators for each attribute

    Example:
        $ datafaker create-generators
    """
    logger.debug("Making %s.", df_file)

    df_file_path = Path(df_file)
    if not force:
        _check_file_non_existence(df_file_path)

    generator_config = read_config_file(config_file) if config_file is not None else {}
    if stats_file is None and generators_require_stats(generator_config):
        stats_file = STATS_FILENAME
    orm_metadata = load_metadata_for_output(orm_file, generator_config)
    result: str = make_table_generators(
        orm_metadata,
        generator_config,
        orm_file,
        config_file,
        stats_file,
    )

    df_file_path.write_text(result, encoding="utf-8")

    logger.debug("%s created.", df_file)


@app.command()
def make_vocab(
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
    config_file: Optional[str] = Option(CONFIG_FILENAME, help="The configuration file"),
    force: bool = Option(
        False,
        "--force/--no-force",
        "-f/+f",
        help="Overwrite any existing vocabulary file.",
    ),
    compress: bool = Option(False, help="Compress file to .gz"),
    only: list[str] = Option([], help="Only download this table."),
) -> None:
    """Make files of vocabulary tables.

    Each table marked in the configuration file as "vocabulary_table: true"

    Example:
        $ datafaker make-vocab --config-file config.yml
    """
    settings = get_settings()
    _require_src_db_dsn(settings)

    generator_config = read_config_file(config_file) if config_file is not None else {}
    orm_metadata = load_metadata(orm_file, generator_config)
    make_vocabulary_tables(
        orm_metadata,
        generator_config,
        overwrite_files=force,
        compress=compress,
        table_names=set(only) if only else None,
    )


@app.command()
def make_stats(
    config_file: Optional[str] = Option(CONFIG_FILENAME, help="The configuration file"),
    stats_file: str = Option(STATS_FILENAME),
    force: bool = Option(
        False, "--force", "-f", help="Overwrite any existing vocabulary file."
    ),
) -> None:
    """Compute summary statistics from the source database.

    Writes the statistics to a YAML file.

    Example:
        $ datafaker make_stats --config-file=example_config.yaml
    """
    logger.debug("Creating %s.", stats_file)

    stats_file_path = Path(stats_file)
    if not force:
        _check_file_non_existence(stats_file_path)

    config = read_config_file(config_file) if config_file is not None else {}

    settings = get_settings()
    src_dsn: str = _require_src_db_dsn(settings)

    src_stats = asyncio.get_event_loop().run_until_complete(
        make_src_stats(src_dsn, config, settings.src_schema)
    )
    stats_file_path.write_text(yaml.dump(src_stats), encoding="utf-8")
    logger.debug("%s created.", stats_file)


@app.command()
def make_tables(
    orm_file: str = Option(ORM_FILENAME, help="Path to write the ORM yaml file to"),
    force: bool = Option(
        False, "--force", "-f", help="Overwrite any existing orm yaml file."
    ),
) -> None:
    """Make a YAML file representing the tables in the schema.

    Example:
        $ datafaker make_tables
    """
    logger.debug("Creating %s.", orm_file)

    orm_file_path = Path(orm_file)
    if not force:
        _check_file_non_existence(orm_file_path)

    settings = get_settings()
    src_dsn: str = _require_src_db_dsn(settings)

    content = make_tables_file(src_dsn, settings.src_schema)
    orm_file_path.write_text(content, encoding="utf-8")
    logger.debug("%s created.", orm_file)


@app.command()
def configure_tables(
    config_file: str = Option(
        CONFIG_FILENAME, help="Path to write the configuration file to"
    ),
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
) -> None:
    """Interactively set tables to ignored, vocabulary or primary private."""
    logger.debug("Configuring tables in %s.", config_file)
    settings = get_settings()
    src_dsn: str = _require_src_db_dsn(settings)
    config_file_path = Path(config_file)
    config = {}
    if config_file_path.exists():
        config = yaml.load(
            config_file_path.read_text(encoding="UTF-8"), Loader=yaml.SafeLoader
        )
    # we don't pass config here so that no tables are ignored
    metadata = load_metadata(orm_file)
    config_updated = update_config_tables(
        src_dsn, settings.src_schema, metadata, config
    )
    if config_updated is None:
        logger.debug("Cancelled")
        return
    content = yaml.dump(config_updated)
    config_file_path.write_text(content, encoding="utf-8")
    logger.debug("Tables configured in %s.", config_file)


@app.command()
def configure_missing(
    config_file: str = Option(
        CONFIG_FILENAME, help="Path to write the configuration file to"
    ),
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
) -> None:
    """Interactively set the missingness of the generated data."""
    logger.debug("Configuring missingness in %s.", config_file)
    settings = get_settings()
    src_dsn: str = _require_src_db_dsn(settings)
    config_file_path = Path(config_file)
    config: dict[str, Any] = {}
    if config_file_path.exists():
        config_any = yaml.load(
            config_file_path.read_text(encoding="UTF-8"), Loader=yaml.SafeLoader
        )
        if isinstance(config_any, dict):
            config = config_any
    metadata = load_metadata(orm_file, config)
    config_updated = update_missingness(src_dsn, settings.src_schema, metadata, config)
    if config_updated is None:
        logger.debug("Cancelled")
        return
    content = yaml.dump(config_updated)
    config_file_path.write_text(content, encoding="utf-8")
    logger.debug("Generators missingness in %s.", config_file)


@app.command()
def configure_generators(
    config_file: str = Option(
        CONFIG_FILENAME, help="Path of the configuration file to alter"
    ),
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
    spec: Path = Option(
        None,
        help=(
            "CSV file (headerless) with fields table-name,"
            " column-name, generator-name to set non-interactively"
        ),
    ),
) -> None:
    """Interactively set generators for column data."""
    logger.debug("Configuring generators in %s.", config_file)
    settings = get_settings()
    src_dsn: str = _require_src_db_dsn(settings)
    config_file_path = Path(config_file)
    config = {}
    if config_file_path.exists():
        config = yaml.load(
            config_file_path.read_text(encoding="UTF-8"), Loader=yaml.SafeLoader
        )
    metadata = load_metadata(orm_file)
    config_updated = update_config_generators(
        src_dsn, settings.src_schema, metadata, config, spec_path=spec
    )
    if config_updated is None:
        logger.debug("Cancelled")
        return
    content = yaml.dump(config_updated)
    config_file_path.write_text(content, encoding="utf-8")
    logger.debug("Generators configured in %s.", config_file)


@app.command()
def dump_data(
    config_file: Optional[str] = Option(
        CONFIG_FILENAME, help="Path of the configuration file to alter"
    ),
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
    table: str = Argument(help="The table to dump"),
    output: str | None = Option(None, help="output CSV file name"),
) -> None:
    """Dump a whole table as a CSV file (or to the console) from the destination database."""
    settings = get_settings()
    dst_dsn: str = settings.dst_dsn or ""
    assert dst_dsn != "", "Missing DST_DSN setting."
    schema_name = settings.dst_schema
    config = read_config_file(config_file) if config_file is not None else {}
    metadata = load_metadata_for_output(orm_file, config)
    if output is None:
        if isinstance(sys.stdout, io.TextIOBase):
            dump_db_tables(metadata, dst_dsn, schema_name, table, sys.stdout)
        return
    with open(output, "wt", newline="", encoding="utf-8") as out:
        dump_db_tables(metadata, dst_dsn, schema_name, table, out)


@app.command()
def validate_config(
    config_file: Path = Argument(help="The configuration file to validate"),
) -> None:
    """Validate the format of a config file."""
    logger.debug("Validating config file: %s.", config_file)

    config = yaml.load(config_file.read_text(encoding="UTF-8"), Loader=yaml.SafeLoader)
    schema_config = json.loads(CONFIG_SCHEMA_PATH.read_text(encoding="UTF-8"))
    try:
        validate(config, schema_config)
    except ValidationError as e:
        logger.error(e)
        sys.exit(1)
    logger.debug("Config file is valid.")


@app.command()
def remove_data(
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
    config_file: Optional[str] = Option(CONFIG_FILENAME, help="The configuration file"),
    yes: bool = Option(
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


@app.command()
def remove_vocab(
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
    config_file: Optional[str] = Option(CONFIG_FILENAME, help="The configuration file"),
    yes: bool = Option(
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


@app.command()
def remove_tables(
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
    config_file: str = Option(CONFIG_FILENAME, help="The configuration file"),
    # pylint: disable=redefined-builtin
    all: bool = Option(
        False,
        help="Don't use the ORM file, delete all tables in the destination schema",
    ),
    yes: bool = Option(
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
            remove_db_tables(metadata)
        logger.debug("Tables dropped.")
    else:
        logger.info("Would remove tables if called with --yes.")


class TableType(str, Enum):
    """Types of tables for the ``list-tables`` command."""

    ALL = "all"
    VOCAB = "vocab"
    GENERATED = "generated"


@app.command()
def list_tables(
    orm_file: str = Option(ORM_FILENAME, help="The name of the ORM yaml file"),
    config_file: Optional[str] = Option(CONFIG_FILENAME, help="The configuration file"),
    tables: TableType = Option(TableType.GENERATED, help="Which tables to list"),
) -> None:
    """List the names of tables described in the metadata file."""
    config = read_config_file(config_file) if config_file is not None else {}
    orm_metadata = load_metadata(orm_file, config)
    all_table_names = set(orm_metadata.tables.keys())
    vocab_table_names = {
        table_name
        for (table_name, table_config) in config.get("tables", {}).items()
        if get_flag(table_config, "vocabulary_table")
    }
    if tables == TableType.ALL:
        names = all_table_names
    elif tables == TableType.GENERATED:
        names = all_table_names - vocab_table_names
    else:
        names = vocab_table_names
    for name in sorted(names):
        print(name)


@app.command()
def version() -> None:
    """Display version information."""
    logger.info(
        "%s version %s",
        __package__,
        importlib.metadata.version(__package__),
    )


if __name__ == "__main__":
    app()
