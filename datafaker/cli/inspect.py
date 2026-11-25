"""
CLI subcommands for inspecting and exporting data from a target synthetic database.

All commands here belong to the "Inspect and Export" rich help panel in the CLI.
"""

import io
import json
import sys
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validate
from sqlalchemy import MetaData, Table
from typer import Argument, Exit, Option

from datafaker.cli._defaults import CONFIG_FILENAME, ORM_FILENAME
from datafaker.cli._load_metadata import load_metadata, load_metadata_for_output
from datafaker.db_utils import generated_tables
from datafaker.dump import (
    CsvTableWriter,
    ParquetTableWriter,
    TableWriter,
    get_parquet_table_writer,
)
from datafaker.settings import get_destination_dsn, get_destination_schema
from datafaker.utils import (
    CONFIG_SCHEMA_PATH,
    get_flag,
    load_yaml_file,
    logger,
    read_config_file,
)


def _convert_table_names_to_tables(
    table_names: list[str], metadata: MetaData
) -> list[Table]:
    """
    Convert a list of table names to SQLAlchemy Tables.

    :param table_names: List of names of tables
    :param metadata: Metadata of the database
    :return: List of tables with names matching ``table_names``
    """
    failed_count = 0
    results: list[Table] = []
    for name in table_names:
        table = metadata.tables.get(name, None)
        if table is not None:
            results.append(table)
        else:
            failed_count += 1
            logger.error(
                "%s is not the name of a table in the destination database", name
            )
    if failed_count:
        raise Exit(1)
    return results


def _dump_csv_to_stdout(
    table: Table,
    metadata: MetaData,
    dsn: str,
    schema_name: str | None,
) -> None:
    """Dump the table to stdout if possible."""
    if isinstance(sys.stdout, io.TextIOBase):
        csv_writer = CsvTableWriter(metadata, dsn, schema_name)
        csv_writer.write_io(table, sys.stdout)


def _get_writer(
    parquet: bool,
    output: Path | None,
    metadata: MetaData,
    dsn: str,
    schema_name: str | None,
) -> TableWriter:
    if parquet or output and output.suffix == ParquetTableWriter.EXTENSION:
        return get_parquet_table_writer(metadata, dsn, schema_name)
    return CsvTableWriter(metadata, dsn, schema_name)


def _dump_tables_to_directory(
    writer: TableWriter,
    directory: Path,
    tables: Iterable[Table],
) -> None:
    count_written = 0
    failed: list[str] = []
    for mtable in tables:
        if writer.write(mtable, directory):
            count_written += 1
        else:
            failed.append(mtable.name)
    logger.info("%d files successfully written", count_written)
    if failed:
        logger.warning("%d files failed:", len(failed))
        for f in failed:
            logger.warning("Failed to write %s", f)


def dump_data(
    config_file: Optional[Path] = Option(
        CONFIG_FILENAME,
        help="Path of the configuration file to use",
        dir_okay=False,
    ),
    orm_file: Path = Option(
        ORM_FILENAME,
        help="The name of the ORM yaml file",
        dir_okay=False,
    ),
    table: list[str] = Option(
        default=[],
        help="The tables to dump (default is all non-ignored, non-vocabulary tables)",
    ),
    output: Path
    | None = Option(
        None,
        help=(
            "Output CSV or Parquet file name,"
            " directory to write into or - to output to the console"
        ),
        file_okay=True,
        dir_okay=True,
    ),
    parquet: bool = Option(
        False,
        help="Use Parquet format (default use CSV unless --output specifies a .parquet file)",
    ),
) -> None:
    """Dump a whole table as a CSV file (or to the console) from the destination database."""
    directory = Path(".")
    if output:
        if Path(output).is_dir():
            directory = Path(output)
            output = None
        elif len(table) != 1:
            logger.error(
                "Must specify exactly one table if the output name is"
                " specified, or specify an existing directory"
            )
            raise Exit(1)
    dst_dsn = get_destination_dsn()
    schema_name = get_destination_schema()
    config = read_config_file(config_file) if config_file is not None else {}
    metadata = load_metadata_for_output(orm_file, config)
    mtables = _convert_table_names_to_tables(table, metadata)
    if not mtables:
        mtables = generated_tables(metadata, config)
    if output is not None and output.name == "-":
        _dump_csv_to_stdout(mtables[0], metadata, dst_dsn, schema_name)
        return
    writer = _get_writer(parquet, output, metadata, dst_dsn, schema_name)
    if output:
        mtable = mtables[0]
        if not writer.write_file(mtable, directory / output):
            logger.error("Could not write table %s to file %s", mtable.name, output)
        return
    _dump_tables_to_directory(writer, directory, mtables)


def validate_config(
    config_file: Path = Argument(help="The configuration file to validate"),
) -> None:
    """Validate the format of a config file."""
    logger.debug("Validating config file: %s.", config_file)

    config = load_yaml_file(config_file, "Config file", "", yaml.SafeLoader)
    schema_config = json.loads(CONFIG_SCHEMA_PATH.read_text(encoding="UTF-8"))
    try:
        validate(config, schema_config)
    except ValidationError as e:
        logger.error(e)
        raise Exit(1) from e
    logger.debug("Config file is valid.")


class TableType(str, Enum):
    """Types of tables for the ``list-tables`` command."""

    ALL = "all"
    VOCAB = "vocab"
    GENERATED = "generated"


def list_tables(
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
