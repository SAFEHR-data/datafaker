"""Entrypoint for the datafaker package."""

import importlib.metadata
import sys
from pathlib import Path
from typing import Optional

from sqlalchemy.exc import InternalError, OperationalError
from typer import Exit, Option, Typer

from datafaker.cli import configure, create, inspect
from datafaker.cli._defaults import CONFIG_FILENAME, ORM_FILENAME
from datafaker.cli._load_metadata import load_metadata_config, load_metadata_for_output
from datafaker.remove import remove_db_data, remove_db_tables, remove_db_vocab
from datafaker.serialize_metadata import dict_to_metadata
from datafaker.settings import SettingsError
from datafaker.theme import ThemeEntry, set_active_theme
from datafaker.utils import conf_logger, logger, read_config_file

# pylint: disable=too-many-arguments


app = Typer(no_args_is_help=True)


def datafaker() -> None:
    """Run the app and catch internal exceptions."""
    try:
        app()
    except OperationalError as exc:
        logger.error(str(exc))
        if (
            type(exc.orig).__module__ == "pyodbc"
            and isinstance(exc.orig, BaseException)
            and 0 < len(exc.orig.args)
            and exc.orig.args[0] == "HYT00"
        ):
            logger.error(
                "Please ensure that the ODBC driver is installed and registered,"
                " and that the database server is available at the location specified."
            )
            logger.error("(see the installation instructions)")
        # Outside of app() typer.Exit(1) doesn't work
        sys.exit(1)
    except SettingsError as exc:
        logger.error(str(exc))
        sys.exit(1)


@app.callback()
def main(
    verbose: bool = Option(False, "--verbose", "-v", help="Print more information."),
    theme: ThemeEntry = Option(
        ThemeEntry.DARK,
        help="The colour scheme",
        envvar="DATAFAKER_THEME",
    ),
) -> None:
    """Set the global parameters."""
    conf_logger(verbose)
    set_active_theme(theme)


app.command(rich_help_panel="Configure and Extract")(configure.make_tables)
app.command(rich_help_panel="Configure and Extract")(configure.make_vocab)
app.command(rich_help_panel="Configure and Extract")(configure.configure_tables)
app.command(rich_help_panel="Configure and Extract")(configure.configure_generators)
app.command(rich_help_panel="Configure and Extract")(configure.configure_missing)
app.command(rich_help_panel="Configure and Extract")(configure.make_stats)

app.command(rich_help_panel="Create Synthetic Database")(create.create_tables)
app.command(rich_help_panel="Create Synthetic Database")(create.create_vocab)
app.command(rich_help_panel="Create Synthetic Database")(create.create_data)

app.command(rich_help_panel="Inspect and Export")(inspect.dump_data)
app.command(rich_help_panel="Inspect and Export")(inspect.validate_config)
app.command(rich_help_panel="Inspect and Export")(inspect.list_tables)


@app.command(rich_help_panel="Remove Destination Data")
def remove_data(
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


@app.command(rich_help_panel="Remove Destination Data")
def remove_vocab(
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


@app.command(rich_help_panel="Remove Destination Data")
def remove_tables(
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
            try:
                remove_db_tables(metadata)
            except InternalError as exc:
                logger.error("Failed to drop tables: %s", exc)
                logger.error("Please try again using the --all option.")
                raise Exit(1) from exc
        logger.debug("Tables dropped.")
    else:
        logger.info("Would remove tables if called with --yes.")


@app.command(rich_help_panel="Inspect and Export")
def version() -> None:
    """Display version information."""
    assert __package__ is not None
    logger.info(
        "%s version %s",
        __package__,
        importlib.metadata.version(__package__),
    )


if __name__ == "__main__":
    datafaker()
