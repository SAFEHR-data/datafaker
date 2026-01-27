"""Add to ORM structure based on parquet files."""
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from fastparquet import ParquetFile

from datafaker.utils import logger


def get_parquet_orm(directory: Path) -> dict[str, Any] | None:
    """
    Read the parquet files, guess a database structure from that.

    :param directory: The directory to search for parquet files in.
    :return: The ORM dictionary on success, None on failure.
    """
    logger.debug("Examining directory %s", directory)
    if not directory.is_dir():
        logger.error("%s is not a directory", directory)
        return None
    parquets: dict[str, ParquetFile] = {}  # type: ignore[no-any-unimported]
    for path, _, filenames in os.walk(directory):
        for filename in filenames:
            entry = Path(path) / filename
            if entry.is_file() and entry.suffix in {".parquet", ".parq"}:
                logger.debug("examining file %s", entry)
                parquets[entry.name] = ParquetFile(entry)
    return {
        name: _get_table_orm(parquet, name, _ColumnGuesser(parquets))
        for name, parquet in parquets.items()
    }


_camel_case_re = re.compile(r"([a-z])([A-Z])")
_word_split_re = re.compile(r"[^A-Za-z0-9]+")


def _get_words(s: str) -> set[str]:
    """Get the words of a string."""
    decamel = _camel_case_re.sub(lambda m: f"{m.group(1)} {m.group(2)}", s)
    return {w.lower() for w in _word_split_re.split(decamel)}


class _ColumnGuesser:
    """Guesses foreign key targets."""

    def __init__(self, tables: dict[str, ParquetFile]) -> None:  # type: ignore[no-any-unimported]
        """Initialize the column guesser."""
        self.tn_cn_words: list[tuple[str, str, set[str]]] = []
        self.tn2words: dict[str, set[str]] = {}
        for name, table in tables.items():
            table_words = _get_words(name)
            self.tn2words[name] = table_words
            for column in table.columns:
                self.tn_cn_words.append(
                    (name, column, table_words | _get_words(column))
                )

    def get_likely_foreign_key_target(
        self,
        column_name: str,
    ) -> tuple[str, str] | None:
        """
        Get a foreign key target string if the name suggests it might be.

        :param column_name: The name of the column which might be a foreign key
        :return: "table.column" or None if no column seems like a likely target
        """
        our_words = _get_words(column_name)
        max_overlaps_so_far = (1, len(our_words) - 1, 0)
        best_so_far: tuple[str, str] | None = None
        for tn, cn, cws in self.tn_cn_words:
            # A triple: (overlap of column name with table name,
            # overlap of column name with column and table name,
            # negative how many words are left over)
            goodness = (
                len(our_words.intersection(self.tn2words[tn])),
                len(our_words.intersection(cws)),
                -len(cws.difference(our_words)),
            )
            # More overlap with table name is better.
            # Same overlap with table name but more overlap with column name is also better.
            if max_overlaps_so_far < goodness:
                max_overlaps_so_far = goodness
                best_so_far = (tn, cn)
        return best_so_far


def _get_table_orm(  # type: ignore[no-any-unimported]
    table: ParquetFile, name: str, column_guesser: _ColumnGuesser
) -> dict[str, Any]:
    """
    Guess the ORM configuration of the table passed in.

    :param table: The parquet of the table to guess the config of.
    :param name: The filename of the parquet file
    :param tables: A dict of parquet file names to parquet files of all the tables.
    :return: A reasonable ORM configuration for this table.
    """
    column_types = {
        column: _dtype_to_sql(dtype) for column, dtype in table.dtypes.items()
    }
    name_pref = name[: name.rfind(".")]
    name_words = _get_words(name_pref)
    cols_orm = {}
    likely_primaries = []
    for column, ctype in column_types.items():
        # A primary key is likely to be the column name plus "_id" or something
        words = _get_words(column) - {"id", "key"}
        col_orm: dict[str, Any] = {}
        if ctype == "INTEGER":
            if words <= name_words:
                likely_primaries.append(column)
                col_orm["primary"] = True
                col_orm["nullable"] = False
            else:
                col_orm["nullable"] = True
            if ctype is not None:
                logger.debug(
                    "Column %s.%s type guessed as %s", name_pref, column, ctype
                )
                col_orm["type"] = ctype
            else:
                logger.warning(
                    "Could not determine type of column %s.%s", name_pref, column
                )
            fk_pair = column_guesser.get_likely_foreign_key_target(column)
            if fk_pair is not None:
                (fk_table, fk_column) = fk_pair
                if fk_table != name:
                    logger.debug(
                        "Column %s.%s guessed as being a foreign key to %s.%s",
                        name_pref,
                        column,
                        fk_table,
                        fk_column,
                    )
                    col_orm["foreign_keys"] = [f"{fk_table}.{fk_column}"]
        cols_orm[column] = col_orm
    if len(likely_primaries) == 0:
        logger.warning("No likely primary keys found for table %s", name)
    elif 1 < len(likely_primaries):
        logger.warning(
            "Found multiple likely primary keys for table %s: %s",
            name,
            likely_primaries,
        )
    return {
        "columns": cols_orm,
        "unique": [],
    }


_numpy_dtype_to_sql: dict[str, str | None] = {
    "?": "BOOLEAN",
    "b": "BOOLEAN",
    "B": "SMALLINT",
    "i": "INTEGER",
    "u": "INTEGER",
    "f": "DOUBLE",
    "M": "DATETIME",
    "U": "TEXT",
    "V": "BLOB",
}


def _dtype_to_sql(dtype: Any) -> str | None:
    """Convert a numpy datatype into a SQL type."""
    if isinstance(dtype, np.dtype):
        if dtype.shape != () or dtype.kind not in _numpy_dtype_to_sql:
            return None
        return _numpy_dtype_to_sql[dtype.kind]
    if isinstance(dtype, str):
        if dtype.startswith("datetime"):
            return "DATETIME"
    return None
