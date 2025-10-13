"""Base configuration command shells."""
import cmd
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any, Optional, Type

import sqlalchemy
from prettytable import PrettyTable
from sqlalchemy import Engine, ForeignKey, MetaData, Table
from typing_extensions import Self

from datafaker.utils import (
    T,
    create_db_engine,
    fk_refers_to_ignored_table,
    get_sync_engine,
)


def or_default(v: T | None, d: T) -> T:
    """Return v if it isn't None, otherwise d."""
    return d if v is None else v


class TableType(Enum):
    """Types of table to be configured."""

    GENERATE = "generate"
    IGNORE = "ignore"
    VOCABULARY = "vocabulary"
    PRIVATE = "private"
    EMPTY = "empty"


TYPE_LETTER = {
    TableType.GENERATE: "G",
    TableType.IGNORE: "I",
    TableType.VOCABULARY: "V",
    TableType.PRIVATE: "P",
    TableType.EMPTY: "e",
}

TYPE_PROMPT = {
    TableType.GENERATE: "(table: {}) ",
    TableType.IGNORE: "(table: {} (ignore)) ",
    TableType.VOCABULARY: "(table: {} (vocab)) ",
    TableType.PRIVATE: "(table: {} (private)) ",
    TableType.EMPTY: "(table: {} (empty))",
}


@dataclass
class TableEntry:
    """Base class for table entries for interactive commands."""

    name: str  # name of the table


class AskSaveCmd(cmd.Cmd):
    """Interactive shell for whether to save and quit."""

    intro = "Do you want to save this configuration?"
    prompt = "(yes/no/cancel) "
    file = None

    def __init__(self) -> None:
        """Initialise a save command."""
        super().__init__()
        self.result = ""

    def do_yes(self, _arg: str) -> bool:
        """Save the new config.yaml."""
        self.result = "yes"
        return True

    def do_no(self, _arg: str) -> bool:
        """Exit without saving."""
        self.result = "no"
        return True

    def do_cancel(self, _arg: str) -> bool:
        """Do not exit."""
        self.result = "cancel"
        return True


def fk_column_name(fk: ForeignKey) -> str:
    """Display name for a foreign key."""
    if fk_refers_to_ignored_table(fk):
        return f"{fk.target_fullname} (ignored)"
    return str(fk.target_fullname)


class DbCmd(ABC, cmd.Cmd):
    """Base class for interactive configuration commands."""

    INFO_NO_MORE_TABLES = "There are no more tables"
    ERROR_ALREADY_AT_START = "Error: Already at the start"
    ERROR_NO_SUCH_TABLE = "Error: '{0}' is not the name of a table in this database"
    ERROR_NO_SUCH_TABLE_OR_COLUMN = "Error: '{0}' is not the name of a table in this database or a column in this table"
    ROW_COUNT_MSG = "Total row count: {}"

    @abstractmethod
    def make_table_entry(
        self, table_name: str, table_config: Mapping
    ) -> TableEntry | None:
        """
        Make a table entry suitable for this interactive command.

        :param name: The name of the table to make an entry for.
        :param table_config: The part of the ``config.yaml`` referring to this table.
        :return: The table entry or None if this table should not be interacted with.
        """

    def __init__(
        self,
        src_dsn: str,
        src_schema: str | None,
        metadata: MetaData,
        config: MutableMapping[str, Any],
    ):
        """Initialise a DbCmd."""
        super().__init__()
        self.config: MutableMapping[str, Any] = config
        self.metadata = metadata
        self._table_entries: list[TableEntry] = []
        tables_config: MutableMapping = config.get("tables", {})
        if not isinstance(tables_config, MutableMapping):
            tables_config = {}
        for name in metadata.tables.keys():
            table_config = tables_config.get(name, {})
            if not isinstance(table_config, MutableMapping):
                table_config = {}
            entry = self.make_table_entry(name, table_config)
            if entry is not None:
                self._table_entries.append(entry)
        self.table_index = 0
        self.engine = create_db_engine(src_dsn, schema_name=src_schema)

    @property
    def sync_engine(self) -> Engine:
        """Get the synchronous version of the engine."""
        return get_sync_engine(self.engine)

    def __enter__(self) -> Self:
        """Enter a ``with`` statement."""
        return self

    def __exit__(
        self,
        _exc_type: Optional[Type[BaseException]],
        _exc_val: Optional[BaseException],
        _exc_tb: Optional[TracebackType],
    ) -> None:
        """Dispose of this object."""
        self.engine.dispose()

    def print(self, text: str, *args: Any, **kwargs: Any) -> None:
        """Print text, formatted with positional and keyword arguments."""
        print(text.format(*args, **kwargs))

    def print_table(
        self, headings: Sequence[str], rows: Sequence[Sequence[Any]]
    ) -> None:
        """
        Print a table.

        :param headings: List of headings for the table.
        :param rows: List of rows of values.
        """
        output = PrettyTable()
        output.field_names = headings
        for row in rows:
            # Hopefully PrettyTable will accept Sequence in the future, not list
            output.add_row(list(row))
        print(output)

    def print_table_by_columns(self, columns: Mapping[str, Sequence[str]]) -> None:
        """
        Print a table.

        :param columns: Dict of column names to the values in the column.
        """
        output = PrettyTable()
        row_count = max([len(col) for col in columns.values()])
        for field_name, data in columns.items():
            output.add_column(field_name, list(data) + [None] * (row_count - len(data)))
        print(output)

    def print_results(self, result: sqlalchemy.CursorResult) -> None:
        """Print the rows resulting from a database query."""
        self.print_table(list(result.keys()), [list(row) for row in result.all()])

    def ask_save(self) -> str:
        """
        Ask the user if they want to save.

        :return: ``yes``, ``no`` or ``cancel``.
        """
        ask = AskSaveCmd()
        ask.cmdloop()
        return ask.result

    @abstractmethod
    def set_prompt(self) -> None:
        """Set the prompt according to the current state."""
        ...

    def _set_table_index(self, index: int) -> bool:
        """
        Move to a different table.

        :param index: Index of the table to move to.
        :return: True if there is a table with such an index to move to.
        """
        if 0 <= index < len(self._table_entries):
            self.table_index = index
            self.set_prompt()
            return True
        return False

    def next_table(self, report: str = "No more tables") -> bool:
        """
        Move to the next table.

        :param report: The text to print if there is no next table.
        :return: True if there is another table to move to.
        """
        if not self._set_table_index(self.table_index + 1):
            self.print(report)
            return False
        return True

    def table_name(self) -> str:
        """Get the name of the current table."""
        return str(self._table_entries[self.table_index].name)

    def table_metadata(self) -> Table:
        """Get the metadata of the current table."""
        return self.metadata.tables[self.table_name()]

    def _get_column_names(self) -> list[str]:
        """Get the names of the current columns."""
        return [col.name for col in self.table_metadata().columns]

    def report_columns(self) -> None:
        """Print information about the current columns."""
        self.print_table(
            ["name", "type", "primary", "nullable", "foreign key"],
            [
                [
                    name,
                    str(col.type),
                    col.primary_key,
                    col.nullable,
                    ", ".join([fk_column_name(fk) for fk in col.foreign_keys]),
                ]
                for name, col in self.table_metadata().columns.items()
            ],
        )

    def get_table_config(self, table_name: str) -> MutableMapping[str, Any]:
        """Get the configuration of the named table."""
        ts = self.config.get("tables", None)
        if not isinstance(ts, MutableMapping):
            return {}
        t = ts.get(table_name)
        return t if isinstance(t, MutableMapping) else {}

    def set_table_config(
        self, table_name: str, config: MutableMapping[str, Any]
    ) -> None:
        """Set the configuration of the named table."""
        ts = self.config.get("tables", None)
        if not isinstance(ts, MutableMapping):
            self.config["tables"] = {table_name: config}
            return
        ts[table_name] = config

    def _remove_prefix_src_stats(self, prefix: str) -> list[MutableMapping[str, Any]]:
        """Remove all source stats with the given prefix from the configuration."""
        src_stats = self.config.get("src-stats", [])
        new_src_stats = []
        for stat in src_stats:
            if not stat.get("name", "").startswith(prefix):
                new_src_stats.append(stat)
        self.config["src-stats"] = new_src_stats
        return new_src_stats

    def get_nonnull_columns(self, table_name: str) -> list[str]:
        """Get the names of the nullable columns in the named table."""
        metadata_table = self.metadata.tables[table_name]
        return [
            str(name)
            for name, column in metadata_table.columns.items()
            if column.nullable
        ]

    def find_entry_index_by_table_name(self, table_name: str) -> int | None:
        """Get the index of the table entry of the named table."""
        return next(
            (
                i
                for i, entry in enumerate(self._table_entries)
                if entry.name == table_name
            ),
            None,
        )

    def _find_entry_by_table_name(self, table_name: str) -> TableEntry | None:
        """Get the table entry of the named table."""
        for e in self._table_entries:
            if e.name == table_name:
                return e
        return None

    def do_counts(self, _arg: str) -> None:
        """Report the column names with the counts of nulls in them."""
        if len(self._table_entries) <= self.table_index:
            return
        table_name = self.table_name()
        nonnull_columns = self.get_nonnull_columns(table_name)
        colcounts = [f", COUNT({nnc}) AS {nnc}" for nnc in nonnull_columns]
        with self.sync_engine.connect() as connection:
            result = connection.execute(
                sqlalchemy.text(
                    f"SELECT COUNT(*) AS row_count{''.join(colcounts)} FROM {table_name}"
                )
            ).first()
            if result is None:
                self.print("Could not count rows in table {0}", table_name)
                return
            row_count = result.row_count
            self.print(self.ROW_COUNT_MSG, row_count)
            self.print_table(
                ["Column", "NULL count"],
                [
                    [name, row_count - count]
                    for name, count in result._mapping.items()
                    if name != "row_count"
                ],
            )

    def do_select(self, arg: str) -> None:
        """Run a select query over the database and show the first 50 results."""
        max_select_rows = 50
        with self.sync_engine.connect() as connection:
            try:
                result = connection.execute(sqlalchemy.text("SELECT " + arg))
            except sqlalchemy.exc.DatabaseError as exc:
                self.print("Failed to execute: {}", exc)
                return
            row_count = result.rowcount
            self.print(self.ROW_COUNT_MSG, row_count)
            if 50 < row_count:
                self.print("Showing the first {} rows", max_select_rows)
            fields = list(result.keys())
            rows = result.fetchmany(max_select_rows)
            self.print_table(fields, rows)

    def do_peek(self, arg: str) -> None:
        """
        View some data from the current table.

        Use 'peek col1 col2 col3' to see a sample of values from
        columns col1, col2 and col3 in the current table.
        Use 'peek' to see a sample of the current column(s).
        Rows that are enitrely null are suppressed.
        """
        max_peek_rows = 25
        if len(self._table_entries) <= self.table_index:
            return
        table_name = self.table_name()
        col_names = arg.split()
        if not col_names:
            col_names = self._get_column_names()
        nonnulls = [cn + " IS NOT NULL" for cn in col_names]
        with self.sync_engine.connect() as connection:
            cols = ",".join(col_names)
            where = "WHERE" if nonnulls else ""
            nonnull = " OR ".join(nonnulls)
            query = sqlalchemy.text(
                f"SELECT {cols} FROM {table_name} {where} {nonnull}"
                f" ORDER BY RANDOM() LIMIT {max_peek_rows}"
            )
            try:
                result = connection.execute(query)
            except Exception as exc:
                self.print(f'SQL query "{query}" caused exception {exc}')
                return
            self.print_table(list(result.keys()), result.fetchmany(max_peek_rows))

    def complete_peek(
        self, text: str, _line: str, _begidx: int, _endidx: int
    ) -> list[str]:
        """Completions for the ``peek`` command."""
        if len(self._table_entries) <= self.table_index:
            return []
        return [
            col for col in self.table_metadata().columns.keys() if col.startswith(text)
        ]
