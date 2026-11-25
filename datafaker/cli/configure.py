import asyncio
from pathlib import Path
from typing import Any, Optional

import yaml
from typer import Exit, Option

from datafaker.cli._defaults import CONFIG_FILENAME, ORM_FILENAME, STATS_FILENAME
from datafaker.cli._load_metadata import load_metadata, load_metadata_config
from datafaker.interactive import update_config_generators, update_missingness
from datafaker.interactive.base import DbCmd
from datafaker.make import make_src_stats, make_tables_file, make_vocabulary_tables
from datafaker.serialize_metadata import dict_to_metadata
from datafaker.settings import get_source_dsn, get_source_schema
from datafaker.utils import load_yaml_file, logger, read_config_file


def _check_file_non_existence(file_path: Path) -> None:
    """Check that a given file does not exist. Exit with an error message if it does."""
    if file_path.exists():
        logger.error("%s should not already exist. Exiting...", file_path)
        raise Exit(1)


def make_tables(
    orm_file: Path = Option(ORM_FILENAME, help="Path to write the ORM yaml file to"),
    force: bool = Option(
        False, "--force", "-f", help="Overwrite any existing orm yaml file."
    ),
    parquet_dir: Optional[Path] = Option(
        None,
        help=(
            "Directory of Parquet files to consider part of the database."
            " This can be useful when using DuckDB."
            " The .parquet files or directories containing partitioned tables"
            " must all be directly inside this named directory."
            " Make sure you check the output!"
        ),
        file_okay=False,
        dir_okay=True,
    ),
    views_to_tables: bool = Option(
        False,
        help=("Generate all views as tables in the synthetic version."),
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

    content = make_tables_file(
        get_source_dsn(),
        get_source_schema(),
        views_to_tables=views_to_tables,
        parquet_dir=parquet_dir,
    )
    orm_file_path.write_text(content, encoding="utf-8")
    logger.debug("%s created.", orm_file)


def make_vocab(
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
    generator_config = read_config_file(config_file) if config_file is not None else {}
    orm_metadata = load_metadata(orm_file, generator_config)
    make_vocabulary_tables(
        orm_metadata,
        generator_config,
        overwrite_files=force,
        compress=compress,
        table_names=set(only) if only else None,
    )


def configure_tables(
    config_file: Path = Option(
        CONFIG_FILENAME,
        help="Path to write the configuration file to",
        dir_okay=False,
    ),
    orm_file: Path = Option(
        ORM_FILENAME,
        help="The name of the ORM yaml file",
        dir_okay=False,
    ),
) -> None:
    """Interactively set tables to ignored, vocabulary or primary private."""
    logger.debug("Configuring tables in %s.", config_file)
    config = {}
    if config_file.exists():
        config = load_yaml_file(config_file, "Config file", "", yaml.SafeLoader)

    # we don't pass config here so that no tables are ignored
    meta_dict = load_metadata_config(orm_file)
    metadata = dict_to_metadata(meta_dict, None)
    config_updated = update_config_tables(
        get_source_dsn(),
        get_source_schema(),
        metadata,
        config,
        Path(meta_dict["parquet-dir"]) if "parquet-dir" in meta_dict else None,
    )
    if config_updated is None:
        logger.debug("Cancelled")
        return
    content = yaml.dump(config_updated)
    config_file.write_text(content, encoding="utf-8")
    logger.debug("Tables configured in %s.", config_file)


def configure_generators(
    config_file: Path = Option(
        CONFIG_FILENAME,
        help="Path of the configuration file to alter",
        dir_okay=False,
    ),
    orm_file: Path = Option(
        ORM_FILENAME,
        help="The name of the ORM yaml file",
        dir_okay=False,
    ),
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
    config = {}
    if config_file.exists():
        config = load_yaml_file(config_file, "Config file", "", yaml.SafeLoader)
    meta_dict = load_metadata_config(orm_file)
    metadata = dict_to_metadata(meta_dict, None)
    config_updated = update_config_generators(
        DbCmd.Settings(
            get_source_dsn(),
            get_source_schema(),
            config,
            metadata,
            meta_dict.get("parquet-dir", None),
        ),
        spec_path=spec,
    )
    if config_updated is None:
        logger.debug("Cancelled")
        return
    content = yaml.dump(config_updated)
    config_file.write_text(content, encoding="utf-8")
    logger.debug("Generators configured in %s.", config_file)


def configure_missing(
    config_file: Path = Option(
        CONFIG_FILENAME,
        help="Path to write the configuration file to",
        dir_okay=False,
    ),
    orm_file: Path = Option(
        ORM_FILENAME,
        help="The name of the ORM yaml file",
        dir_okay=False,
    ),
) -> None:
    """Interactively set the missingness of the generated data."""
    logger.debug("Configuring missingness in %s.", config_file)
    config: dict[str, Any] = {}
    if config_file.exists():
        config_any = load_yaml_file(config_file, "Config file", "", yaml.SafeLoader)
        if isinstance(config_any, dict):
            config = config_any
    meta_dict = load_metadata_config(orm_file, config)
    metadata = dict_to_metadata(meta_dict, None)
    config_updated = update_missingness(
        get_source_dsn(),
        get_source_schema(),
        metadata,
        config,
        Path(meta_dict["parquet-dir"]) if "parquet-dir" in meta_dict else None,
    )
    if config_updated is None:
        logger.debug("Cancelled")
        return
    content = yaml.dump(config_updated)
    config_file.write_text(content, encoding="utf-8")
    logger.debug("Generators missingness in %s.", config_file)


def make_stats(
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
    stats_file: Path = Option(STATS_FILENAME),
    force: bool = Option(
        False, "--force", "-f", help="Overwrite any existing vocabulary file."
    ),
) -> None:
    """Compute summary statistics from the source database."""
    logger.debug("Creating %s.", stats_file)

    if not force:
        _check_file_non_existence(stats_file)

    config = read_config_file(config_file) if config_file is not None else {}
    meta_dict = load_metadata_config(orm_file, config)

    src_stats = asyncio.get_event_loop().run_until_complete(
        make_src_stats(
            get_source_dsn(),
            config,
            get_source_schema(),
            parquet_dir=meta_dict.get("parquet-dir", None),
        )
    )
    stats_file.write_text(yaml.dump(src_stats), encoding="utf-8")
    logger.debug("%s created.", stats_file)
