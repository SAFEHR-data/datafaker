from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import MetaData
from typer import Exit

from datafaker.serialize_metadata import dict_to_metadata, should_ignore_fk
from datafaker.utils import get_flag, load_yaml_file, logger


def load_metadata_config(
    orm_file_path: Path, config: dict | None = None
) -> dict[str, Any]:
    """
    Load the ``orm.yaml`` file, returning a dict representation.

    :param orm_file_name: The name of the file to load.
    :param config: The ``config.yaml`` file object. Ignored tables will be
        excluded from the output.
    :return: A dict representing the ``orm.yaml`` file, with the tables
        the ``config`` says to ignore removed.
    """
    meta_dict = load_yaml_file(
        orm_file_path,
        "ORM file",
        "Please run make-tables first.",
        yaml.SafeLoader,
    )
    if not isinstance(meta_dict, dict):
        logger.error(
            "ORM file '%s' is empty or invalid. Please run make-tables --force to recreate it.",
            orm_file_path,
        )
        raise Exit(1)
    tables_dict = meta_dict.get("tables", {})
    if config is not None and "tables" in config:
        tables_config = config["tables"]
        # Remove ignored tables
        for name, table_config in tables_config.items():
            if get_flag(table_config, "ignore"):
                tables_dict.pop(name, None)
        # Remove foreign keys to ignored tables
        for table_meta in tables_dict.values():
            for column_meta in table_meta.get("columns", {}).values():
                if isinstance(column_meta, dict) and "foreign_keys" in column_meta:
                    filtered_fks = [
                        fk
                        for fk in column_meta["foreign_keys"]
                        if not should_ignore_fk(tables_config, fk)
                    ]
                    column_meta["foreign_keys"] = filtered_fks
    return meta_dict


def load_metadata(orm_file_path: Path, config: dict | None = None) -> MetaData:
    """
    Load metadata from ``orm.yaml``.

    :param orm_file_name: ``orm.yaml`` or alternative name to load metadata from.
    :param config: Used to exclude tables that are marked as ``ignore: true``.
    :return: SQLAlchemy MetaData object representing the database described by the loaded file.
    """
    meta_dict = load_metadata_config(orm_file_path, config)
    return dict_to_metadata(meta_dict, None)


def load_metadata_for_output(
    orm_file_path: Path, config: dict | None = None
) -> MetaData:
    """Load metadata excluding any foreign keys pointing to ignored tables."""
    meta_dict = load_metadata_config(orm_file_path, config)
    return dict_to_metadata(meta_dict, config)
