"""Add to ORM structure based on parquet files."""
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastparquet import ParquetFile
from typer import Exit

from datafaker.utils import logger

PARQUET_SUFFIXES = {".parquet", ".parq"}

# The dtype recorded for a Hive partition key (such as the ``yr`` in ``yr=2023``).
# Partition keys are not stored in the files themselves.
PARTITION_KEY_DTYPE = "category"

_hive_dir_re = re.compile(r"^([^=]+)=(.*)$")


@dataclass(frozen=True)
class ParquetTable:
    """A table found in a directory of parquet files."""

    name: str
    """The name of the table, as it will appear in ``orm.yaml``."""
    root: Path
    """The parquet file for a single-file table, or the directory of a dataset."""
    files: tuple[Path, ...]
    """All the parquet files that make up the table, sorted."""
    is_dataset: bool
    """True if the table is a (possibly partitioned) directory of parquet files."""


def _parquet_files_under(directory: Path) -> tuple[Path, ...]:
    """Find all the parquet files in ``directory`` and below, sorted."""
    return tuple(
        sorted(
            entry
            for entry in directory.rglob("*")
            if entry.suffix in PARQUET_SUFFIXES and entry.is_file()
        )
    )


def find_parquet_tables(directory: Path) -> list[ParquetTable]:
    """
    Find the tables represented by the parquet files in a directory.

    * A parquet file directly in ``directory`` is a table named after the file.
    * A subdirectory of ``directory`` containing parquet files at any depth
      is a single (partitioned) table named after the subdirectory.
    * If ``directory`` has no parquet files directly in it, and all the
      subdirectories containing parquet files are Hive partitions
      (named ``key=value``), then ``directory`` is itself a single table.

    :param directory: The directory to search for parquet files in.
    :return: The tables found, files first, then datasets, each sorted by name.
    """
    directory = Path(directory)
    children = sorted(directory.iterdir())
    file_tables = [
        ParquetTable(entry.name, entry, (entry,), False)
        for entry in children
        if entry.suffix in PARQUET_SUFFIXES and entry.is_file()
    ]
    dir_files = [
        (entry, files)
        for entry in children
        if entry.is_dir()
        for files in [_parquet_files_under(entry)]
        if files
    ]
    if not file_tables and dir_files:
        if all(_hive_dir_re.match(entry.name) for entry, _ in dir_files):
            name = directory.resolve().name
            return [
                ParquetTable(
                    name,
                    directory,
                    tuple(f for _, files in dir_files for f in files),
                    True,
                )
            ]
    return file_tables + [
        ParquetTable(entry.name, entry, files, True) for entry, files in dir_files
    ]


def _file_dtypes(path: Path) -> dict[str, Any]:
    """Get the column names and dtypes stored in one parquet file."""
    return dict(ParquetFile(path).dtypes)


def _partition_keys(table: ParquetTable, path: Path) -> list[str]:
    """Get the Hive partition keys of one file of a dataset, in path order."""
    keys = []
    for part in path.relative_to(table.root).parts[:-1]:
        match = _hive_dir_re.match(part)
        if match:
            keys.append(match.group(1))
    return keys


def _describe_column_difference(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> str:
    """Describe how the column names of ``actual`` differ from ``expected``."""
    problems = []
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    if missing:
        problems.append(f"missing columns {missing}")
    if extra:
        problems.append(f"extra columns {extra}")
    return "; ".join(problems)


def _type_name(dtype: Any) -> str:
    """Get a name for a dtype that ignores case, so ``int32`` equals ``Int32``."""
    return str(dtype).lower()


def get_table_dtypes(table: ParquetTable) -> dict[str, Any]:
    """
    Get the columns and dtypes of a table, checking that its files agree.

    All the files of a partitioned table must have the same column names and
    the same partition keys. Directories that contain unrelated tables are
    rejected with an error message. Column types may differ between files
    (for example when a partition has no values for a column); this is
    reported as a warning and the type from the first file is used.

    :param table: The table to examine.
    :return: A dict of column names to dtypes. Partition keys are included.
    :raises typer.Exit: If the files of a dataset do not all have the same columns.
    """
    first = table.files[0]
    dtypes = _file_dtypes(first)
    if not table.is_dataset:
        return dtypes
    keys = _partition_keys(table, first)
    type_names: dict[str, set[str]] = {}
    for path in table.files[1:]:
        file_dtypes = _file_dtypes(path)
        problem = _describe_column_difference(dtypes, file_dtypes)
        if not problem and _partition_keys(table, path) != keys:
            problem = "different partition directories"
        if problem:
            logger.error(
                "Directory %s cannot be read as a single partitioned table: "
                "%s differs from %s (%s). All the files in a partitioned "
                "table must have the same columns. If the "
                "directories inside %s hold different tables, move each "
                "table into its own directory directly inside the parquet "
                "directory and run this command again.",
                table.name,
                path,
                first,
                problem,
                table.root,
            )
            raise Exit(1)
        for column, dtype in file_dtypes.items():
            if _type_name(dtype) != _type_name(dtypes[column]):
                names = type_names.setdefault(column, {_type_name(dtypes[column])})
                names.add(_type_name(dtype))
    for column, names in type_names.items():
        logger.warning(
            "Column %s.%s has different types in different files: %s. "
            "Using the type from %s.",
            table.name,
            column,
            sorted(names),
            first,
        )
    return {**dtypes, **{k: PARTITION_KEY_DTYPE for k in keys if k not in dtypes}}


def get_parquet_orm(directory: Path) -> dict[str, Any] | None:
    """
    Read the parquet files, guess a database structure from that.

    A directory of parquet files (possibly partitioned) is read as one table.

    :param directory: The directory to search for parquet files in.
    :return: The ORM dictionary on success, None on failure.
    :raises typer.Exit: If a directory holds files with differing schemas.
    """
    logger.debug("Examining directory %s", directory)
    if not directory.is_dir():
        logger.error("%s is not a directory", directory)
        return None
    tables = find_parquet_tables(directory)
    dtypes = {table.name: get_table_dtypes(table) for table in tables}
    guesser = _ColumnGuesser(dtypes)
    return {
        name: _get_table_orm(table_dtypes, name, guesser)
        for name, table_dtypes in dtypes.items()
    }


_camel_case_re = re.compile(r"([a-z])([A-Z])")
_word_split_re = re.compile(r"[^A-Za-z0-9]+")


def _get_words(s: str) -> set[str]:
    """Get the words of a string."""
    decamel = _camel_case_re.sub(lambda m: f"{m.group(1)} {m.group(2)}", s)
    return {w.lower() for w in _word_split_re.split(decamel)}


class _ColumnGuesser:
    """Guesses foreign key targets."""

    def __init__(self, tables: Mapping[str, Iterable[str]]) -> None:
        """Initialize the column guesser from table names to column names."""
        self.tn_cn_words: list[tuple[str, str, set[str]]] = []
        self.tn2words: dict[str, set[str]] = {}
        for name, table in tables.items():
            table_words = _get_words(name)
            self.tn2words[name] = table_words
            for column in table:
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


def _get_table_orm(
    dtypes: Mapping[str, Any], name: str, column_guesser: _ColumnGuesser
) -> dict[str, Any]:
    """
    Guess the ORM configuration of the table passed in.

    :param dtypes: The column names and dtypes of the table to guess the config of.
    :param name: The filename of the parquet file, or directory name of the dataset.
    :param column_guesser: Guesses foreign key targets across all the tables.
    :return: A reasonable ORM configuration for this table.
    """
    column_types = {column: _dtype_to_sql(dtype) for column, dtype in dtypes.items()}
    # Get everything before the last dot
    # (or the whole thing if there is no dot)
    name_pref = name.rsplit(".", 1)[0]
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
            fk_pair = column_guesser.get_likely_foreign_key_target(column)
            if fk_pair is not None:
                fk_table, fk_column = fk_pair
                if fk_table != name:
                    logger.debug(
                        "Column %s.%s guessed as being a foreign key to %s.%s",
                        name_pref,
                        column,
                        fk_table,
                        fk_column,
                    )
                    col_orm["foreign_keys"] = [f"{fk_table}.{fk_column}"]
        if ctype is not None:
            logger.debug("Column %s.%s type guessed as %s", name_pref, column, ctype)
            col_orm["type"] = ctype
        else:
            logger.warning(
                "Could not determine type of column %s.%s", name_pref, column
            )
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
    # O can be anything, but Pandas before version 3 makes text O
    "O": "TEXT",
    "U": "TEXT",
    "V": "BLOB",
}


def _dtype_to_sql(dtype: Any) -> str | None:
    """Convert a numpy or Pandas datatype into a SQL type."""
    # Pandas nullable types such as ``Int32`` wrap a numpy dtype
    dtype = getattr(dtype, "numpy_dtype", dtype)
    if isinstance(dtype, np.dtype):
        if dtype.shape != () or dtype.kind not in _numpy_dtype_to_sql:
            return None
        return _numpy_dtype_to_sql[dtype.kind]
    if isinstance(dtype, str):
        if dtype.startswith("datetime"):
            return "DATETIME"
        if dtype == PARTITION_KEY_DTYPE:
            return "TEXT"
    return None
