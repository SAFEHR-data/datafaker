"""Interactive configuration commands."""
import cmd
import csv
import functools
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Iterable, Optional, Type, cast

import sqlalchemy
from prettytable import PrettyTable
from sqlalchemy import Column, Engine, ForeignKey, MetaData, Table, text
from typing_extensions import Self

from datafaker.generators import Generator, PredefinedGenerator, everything_factory
from datafaker.utils import (
    T,
    create_db_engine,
    fk_refers_to_ignored_table,
    get_sync_engine,
    logger,
    primary_private_fks,
    table_is_private,
)

# Monkey patch pyreadline3 v3.5 so that it works with Python 3.13
# Windows users can install pyreadline3 to get tab completion working.
# See https://github.com/pyreadline3/pyreadline3/issues/37
try:
    import readline

    if not hasattr(readline, "backend"):
        setattr(readline, "backend", "readline")
except:
    pass


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
    def make_table_entry(self, name: str, table_config: Mapping) -> TableEntry | None:
        """
        Make a table entry suitable for this interactive command.

        :param name: The name of the table to make an entry for.
        :param table_config: The part of the ``config.yaml`` referring to this table.
        :return: The table entry or None if this table should not be interacted with.
        """
        ...

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
        tables_config: Mapping = config.get("tables", {})
        if type(tables_config) is not dict:
            tables_config = {}
        for name in metadata.tables.keys():
            table_config = tables_config.get(name, {})
            if type(table_config) is not dict:
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

    def print_table(self, headings: list[str], rows: list[list[Any]]) -> None:
        """
        Print a table.

        :param headings: List of headings for the table.
        :param rows: List of rows of values.
        """
        output = PrettyTable()
        output.field_names = headings
        for row in rows:
            output.add_row(row)
        print(output)

    def print_table_by_columns(self, columns: dict[str, list[str]]) -> None:
        """
        Print a table.

        :param columns: Dict of column names to the values in the column.
        """
        output = PrettyTable()
        row_count = max([len(col) for col in columns.values()])
        for field_name, data in columns.items():
            output.add_column(field_name, data + [None] * (row_count - len(data)))
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

    def set_table_index(self, index: int) -> bool:
        """
        Move to a different table.

        :param index: Index of the table to move to.
        :return: True if there is a table with such an index to move to.
        """
        if 0 <= index and index < len(self._table_entries):
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
        if not self.set_table_index(self.table_index + 1):
            self.print(report)
            return False
        return True

    def table_name(self) -> str:
        """Get the name of the current table."""
        return str(self._table_entries[self.table_index].name)

    def table_metadata(self) -> Table:
        """Get the metadata of the current table."""
        return self.metadata.tables[self.table_name()]

    def get_column_names(self) -> list[str]:
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

    def get_table_config(self, table_name: str) -> dict[str, Any]:
        """Get the configuration of the named table."""
        ts = self.config.get("tables", None)
        if type(ts) is not dict:
            return {}
        t = ts.get(table_name)
        return t if type(t) is dict else {}

    def set_table_config(self, table_name: str, config: dict[str, Any]) -> None:
        """Set the configuration of the named table."""
        ts = self.config.get("tables", None)
        if type(ts) is not dict:
            self.config["tables"] = {table_name: config}
            return
        ts[table_name] = config

    def _remove_prefix_src_stats(self, prefix: str) -> list[dict[str, Any]]:
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

    def find_entry_by_table_name(self, table_name: str) -> TableEntry | None:
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
        colcounts = [", COUNT({0}) AS {0}".format(nnc) for nnc in nonnull_columns]
        with self.sync_engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT COUNT(*) AS row_count{colcounts} FROM {table}".format(
                        table=table_name,
                        colcounts="".join(colcounts),
                    )
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
        MAX_SELECT_ROWS = 50
        with self.sync_engine.connect() as connection:
            try:
                result = connection.execute(text("SELECT " + arg))
            except sqlalchemy.exc.DatabaseError as exc:
                self.print("Failed to execute: {}", exc)
                return
            row_count = result.rowcount
            self.print(self.ROW_COUNT_MSG, row_count)
            if 50 < row_count:
                self.print("Showing the first {} rows", MAX_SELECT_ROWS)
            fields = list(result.keys())
            rows = [row._tuple() for row in result.fetchmany(MAX_SELECT_ROWS)]
            self.print_table(fields, rows)

    def do_peek(self, arg: str) -> None:
        """
        View some data from the current table.

        Use 'peek col1 col2 col3' to see a sample of values from columns col1, col2 and col3 in the current table.
        Use 'peek' to see a sample of the current column(s).
        Rows that are enitrely null are suppressed.
        """
        MAX_PEEK_ROWS = 25
        if len(self._table_entries) <= self.table_index:
            return
        table_name = self.table_name()
        col_names = arg.split()
        if not col_names:
            col_names = self.get_column_names()
        nonnulls = [cn + " IS NOT NULL" for cn in col_names]
        with self.sync_engine.connect() as connection:
            query = "SELECT {cols} FROM {table} {where} {nonnull} ORDER BY RANDOM() LIMIT {max}".format(
                cols=",".join(col_names),
                table=table_name,
                where="WHERE" if nonnulls else "",
                nonnull=" OR ".join(nonnulls),
                max=MAX_PEEK_ROWS,
            )
            try:
                result = connection.execute(text(query))
            except Exception as exc:
                self.print(f'SQL query "{query}" caused exception {exc}')
                return
            rows = [row._tuple() for row in result.fetchmany(MAX_PEEK_ROWS)]
            self.print_table(list(result.keys()), rows)

    def complete_peek(
        self, text: str, _line: str, _begidx: int, _endidx: int
    ) -> list[str]:
        """Completions for the ``peek`` command."""
        if len(self._table_entries) <= self.table_index:
            return []
        return [
            col for col in self.table_metadata().columns.keys() if col.startswith(text)
        ]


@dataclass
class TableCmdTableEntry(TableEntry):
    """Table entry for the table command shell."""

    old_type: TableType
    new_type: TableType


class TableCmd(DbCmd):
    """Command shell allowing the user to set the type of each table."""

    intro = "Interactive table configuration (ignore, vocabulary, private, generate or empty). Type ? for help.\n"
    doc_leader = """Use the commands 'ignore', 'vocabulary',
'private', 'empty' or 'generate' to set the table's type. Use 'next' or
'previous' to change table. Use 'tables' and 'columns' for
information about the database. Use 'data', 'peek', 'select' or
'count' to see some data contained in the current table. Use 'quit'
to exit this program."""
    prompt = "(tableconf) "
    file = None
    WARNING_TEXT_VOCAB_TO_NON_VOCAB = (
        "Vocabulary table {0} references non-vocabulary table {1}"
    )
    WARNING_TEXT_NON_EMPTY_TO_EMPTY = (
        "Empty table {1} referenced from non-empty table {0}. {1} will need stories."
    )
    WARNING_TEXT_PROBLEMS_EXIST = "WARNING: The following table types have problems:"
    WARNING_TEXT_POTENTIAL_PROBLEMS = (
        "NOTE: The following table types might cause problems later:"
    )
    NOTE_TEXT_NO_CHANGES = "You have made no changes."
    NOTE_TEXT_CHANGING = "Changing {0} from {1} to {2}"

    def make_table_entry(self, name: str, table: Mapping) -> TableCmdTableEntry | None:
        """
        Make a table entry for the named table.

        :param name: The name of the table.
        :param table: The part of ``config.yaml`` corresponding to this table.
        :return: The newly-constructed table entry.
        """
        if table.get("ignore", False):
            return TableCmdTableEntry(name, TableType.IGNORE, TableType.IGNORE)
        if table.get("vocabulary_table", False):
            return TableCmdTableEntry(name, TableType.VOCABULARY, TableType.VOCABULARY)
        if table.get("primary_private", False):
            return TableCmdTableEntry(name, TableType.PRIVATE, TableType.PRIVATE)
        if table.get("num_rows_per_pass", 1) == 0:
            return TableCmdTableEntry(name, TableType.EMPTY, TableType.EMPTY)
        return TableCmdTableEntry(name, TableType.GENERATE, TableType.GENERATE)

    def __init__(
        self,
        src_dsn: str,
        src_schema: str | None,
        metadata: MetaData,
        config: MutableMapping[str, Any],
    ) -> None:
        """Initialise a TableCmd."""
        super().__init__(src_dsn, src_schema, metadata, config)
        self.set_prompt()

    @property
    def table_entries(self) -> list[TableCmdTableEntry]:
        """Get the list of table entries."""
        return cast(list[TableCmdTableEntry], self._table_entries)

    def find_entry_by_table_name(self, table_name: str) -> TableCmdTableEntry | None:
        """Get the table entry of the table with the given name."""
        entry = super().find_entry_by_table_name(table_name)
        if entry is None:
            return None
        return cast(TableCmdTableEntry, entry)

    def set_prompt(self) -> None:
        """Set the prompt according to the current table and its type."""
        if self.table_index < len(self.table_entries):
            entry = self.table_entries[self.table_index]
            self.prompt = TYPE_PROMPT[entry.new_type].format(entry.name)
        else:
            self.prompt = "(table) "

    def set_type(self, t_type: TableType) -> None:
        """Set the type of the current table."""
        if self.table_index < len(self.table_entries):
            entry = self.table_entries[self.table_index]
            entry.new_type = t_type

    def _copy_entries(self) -> None:
        """Alter the configuration to match the new table entries."""
        for entry in self.table_entries:
            if entry.old_type != entry.new_type:
                table = self.get_table_config(entry.name)
                if (
                    entry.old_type == TableType.EMPTY
                    and table.get("num_rows_per_pass", 1) == 0
                ):
                    table["num_rows_per_pass"] = 1
                if entry.new_type == TableType.IGNORE:
                    table["ignore"] = True
                    table.pop("vocabulary_table", None)
                    table.pop("primary_private", None)
                elif entry.new_type == TableType.VOCABULARY:
                    table.pop("ignore", None)
                    table["vocabulary_table"] = True
                    table.pop("primary_private", None)
                elif entry.new_type == TableType.PRIVATE:
                    table.pop("ignore", None)
                    table.pop("vocabulary_table", None)
                    table["primary_private"] = True
                elif entry.new_type == TableType.EMPTY:
                    table.pop("ignore", None)
                    table.pop("vocabulary_table", None)
                    table.pop("primary_private", None)
                    table["num_rows_per_pass"] = 0
                else:
                    table.pop("ignore", None)
                    table.pop("vocabulary_table", None)
                    table.pop("primary_private", None)
                self.set_table_config(entry.name, table)

    def _get_referenced_tables(self, from_table_name: str) -> set[str]:
        """Get all the tables referenced by this table's foreign keys."""
        from_meta = self.metadata.tables[from_table_name]
        return {
            fk.column.table.name for col in from_meta.columns for fk in col.foreign_keys
        }

    def _sanity_check_failures(self) -> list[tuple[str, str, str]]:
        """Find tables that reference each other that should not given their types."""
        failures = []
        for from_entry in self.table_entries:
            from_t = from_entry.new_type
            if from_t == TableType.VOCABULARY:
                referenced = self._get_referenced_tables(from_entry.name)
                for ref in referenced:
                    to_entry = self.find_entry_by_table_name(ref)
                    if (
                        to_entry is not None
                        and to_entry.new_type != TableType.VOCABULARY
                    ):
                        failures.append(
                            (
                                self.WARNING_TEXT_VOCAB_TO_NON_VOCAB,
                                from_entry.name,
                                to_entry.name,
                            )
                        )
        return failures

    def _sanity_check_warnings(self) -> list[tuple[str, str, str]]:
        """Find tables that reference each other that might cause problems given their types."""
        warnings = []
        for from_entry in self.table_entries:
            from_t = from_entry.new_type
            if from_t in {TableType.GENERATE, TableType.PRIVATE}:
                referenced = self._get_referenced_tables(from_entry.name)
                for ref in referenced:
                    to_entry = self.find_entry_by_table_name(ref)
                    if to_entry is not None and to_entry.new_type in {
                        TableType.EMPTY,
                        TableType.IGNORE,
                    }:
                        warnings.append(
                            (
                                self.WARNING_TEXT_NON_EMPTY_TO_EMPTY,
                                from_entry.name,
                                to_entry.name,
                            )
                        )
        return warnings

    def do_quit(self, _arg: str) -> bool:
        """Check the updates, save them if desired and quit the configurer."""
        count = 0
        for entry in self.table_entries:
            if entry.old_type != entry.new_type:
                count += 1
                self.print(
                    self.NOTE_TEXT_CHANGING,
                    entry.name,
                    entry.old_type.value,
                    entry.new_type.value,
                )
        if count == 0:
            self.print(self.NOTE_TEXT_NO_CHANGES)
        failures = self._sanity_check_failures()
        if failures:
            self.print(self.WARNING_TEXT_PROBLEMS_EXIST)
            for text, from_t, to_t in failures:
                self.print(text, from_t, to_t)
        warnings = self._sanity_check_warnings()
        if warnings:
            self.print(self.WARNING_TEXT_POTENTIAL_PROBLEMS)
            for text, from_t, to_t in warnings:
                self.print(text, from_t, to_t)
        reply = self.ask_save()
        if reply == "yes":
            self._copy_entries()
            return True
        if reply == "no":
            return True
        return False

    def do_tables(self, _arg: str) -> None:
        """List the tables with their types."""
        for entry in self.table_entries:
            old = entry.old_type
            new = entry.new_type
            becomes = "   " if old == new else "->" + TYPE_LETTER[new]
            self.print("{0}{1} {2}", TYPE_LETTER[old], becomes, entry.name)

    def do_next(self, arg: str) -> None:
        """'next' = go to the next table, 'next tablename' = go to table 'tablename'."""
        if arg:
            # Find the index of the table called _arg, if any
            index = self.find_entry_index_by_table_name(arg)
            if index is None:
                self.print(self.ERROR_NO_SUCH_TABLE, arg)
                return
            self.set_table_index(index)
            return
        self.next_table(self.INFO_NO_MORE_TABLES)

    def complete_next(
        self, text: str, _line: str, _begidx: int, _endidx: int
    ) -> list[str]:
        """Get the completions for tables and columns."""
        return [
            entry.name for entry in self.table_entries if entry.name.startswith(text)
        ]

    def do_previous(self, _arg: str) -> None:
        """Go to the previous table."""
        if not self.set_table_index(self.table_index - 1):
            self.print(self.ERROR_ALREADY_AT_START)

    def do_ignore(self, _arg: str) -> None:
        """Set the current table as ignored, and go to the next table."""
        self.set_type(TableType.IGNORE)
        self.print("Table {} set as ignored", self.table_name())
        self.next_table()

    def do_vocabulary(self, _arg: str) -> None:
        """Set the current table as a vocabulary table, and go to the next table."""
        self.set_type(TableType.VOCABULARY)
        self.print("Table {} set to be a vocabulary table", self.table_name())
        self.next_table()

    def do_private(self, _arg: str) -> None:
        """Set the current table as a primary private table (such as the table of patients)."""
        self.set_type(TableType.PRIVATE)
        self.print("Table {} set to be a primary private table", self.table_name())
        self.next_table()

    def do_generate(self, _arg: str) -> None:
        """Set the current table as neither a vocabulary table nor ignored nor primary private, and go to the next table."""
        self.set_type(TableType.GENERATE)
        self.print("Table {} generate", self.table_name())
        self.next_table()

    def do_empty(self, _arg: str) -> None:
        """Set the current table as empty; no generators will be run for it."""
        self.set_type(TableType.EMPTY)
        self.print("Table {} empty", self.table_name())
        self.next_table()

    def do_columns(self, _arg: str) -> None:
        """Report the column names and metadata."""
        self.report_columns()

    def do_data(self, arg: str) -> None:
        """
        Report some data.

        'data' = report a random ten lines,
        'data 20' = report a random 20 lines,
        'data 20 ColumnName' = report a random twenty entries from ColumnName,
        'data 20 ColumnName 30' = report a random twenty entries from ColumnName of length at least 30,
        """
        args = arg.split()
        column = None
        number = None
        arg_index = 0
        min_length = 0
        table_metadata = self.table_metadata()
        if arg_index < len(args) and args[arg_index].isdigit():
            number = int(args[arg_index])
            arg_index += 1
        if arg_index < len(args) and args[arg_index] in table_metadata.columns:
            column = args[arg_index]
            arg_index += 1
        if arg_index < len(args) and args[arg_index].isdigit():
            min_length = int(args[arg_index])
            arg_index += 1
        if arg_index != len(args):
            self.print(
                """Did not understand these arguments
The format is 'data [entries] [column-name [minimum-length]]' where [] means optional text.
Type 'columns' to find out valid column names for this table.
Type 'help data' for examples."""
            )
            return
        if column is None:
            if number is None:
                number = 10
            self.print_row_data(number)
        else:
            if number is None:
                number = 48
            self.print_column_data(column, number, min_length)

    def complete_data(
        self, text: str, line: str, begidx: int, _endidx: int
    ) -> list[str]:
        """Get completions for arguments to ``data``."""
        previous_parts = line[: begidx - 1].split()
        if len(previous_parts) != 2:
            return []
        table_metadata = self.table_metadata()
        return [k for k in table_metadata.columns.keys() if k.startswith(text)]

    def print_column_data(self, column: str, count: int, min_length: int) -> None:
        """
        Print a sample of data from a certain column of the current table.

        :param column: The name of the column to report on.
        :param count: The number of rows to sample.
        :param min_length: The minimum length of text to choose from (0 for any text).
        """
        where = f"WHERE {column} IS NOT NULL"
        if 0 < min_length:
            where = "WHERE LENGTH({column}) >= {len}".format(
                column=column,
                len=min_length,
            )
        with self.sync_engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT {column} FROM {table} {where} ORDER BY RANDOM() LIMIT {count}".format(
                        table=self.table_name(),
                        column=column,
                        count=count,
                        where=where,
                    )
                )
            )
            self.columnize([str(x[0]) for x in result.all()])

    def print_row_data(self, count: int) -> None:
        """
        Print a sample or rows from the current table.

        :param count: The number of rows to report.
        """
        with self.sync_engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT * FROM {table} ORDER BY RANDOM() LIMIT {count}".format(
                        table=self.table_name(),
                        count=count,
                    )
                )
            )
            if result is None:
                self.print("No rows in this table!")
                return
            self.print_results(result)


def update_config_tables(
    src_dsn: str, src_schema: str | None, metadata: MetaData, config: MutableMapping
) -> Mapping[str, Any]:
    """Ask the user to specify what should happen to each table."""
    with TableCmd(src_dsn, src_schema, metadata, config) as tc:
        tc.cmdloop()
        return tc.config


@dataclass
class MissingnessType:
    """The functions required for applying missingness."""

    SAMPLED = "column_presence.sampled"
    SAMPLED_QUERY = (
        "SELECT COUNT(*) AS row_count, {result_names} FROM "
        "(SELECT {column_is_nulls} FROM {table} ORDER BY RANDOM() LIMIT {count})"
        " AS __t GROUP BY {result_names}"
    )
    name: str
    query: str
    comment: str
    columns: list[str]

    @classmethod
    def sampled_query(cls, table: str, count: int, column_names: Iterable[str]) -> str:
        """
        Construct a query to make a sampling of the named rows of the table.

        :param table: The name of the table to sample.
        :param count: The number of samples to get.
        :param column_names: The columns to fetch.
        :return: The SQL query to do the sampling.
        """
        result_names = ", ".join(["{0}__is_null".format(c) for c in column_names])
        column_is_nulls = ", ".join(
            ["{0} IS NULL AS {0}__is_null".format(c) for c in column_names]
        )
        return cls.SAMPLED_QUERY.format(
            result_names=result_names,
            column_is_nulls=column_is_nulls,
            table=table,
            count=count,
        )


@dataclass
class MissingnessCmdTableEntry(TableEntry):
    """Table entry for the missingness command shell."""

    old_type: MissingnessType
    new_type: MissingnessType | None


class MissingnessCmd(DbCmd):
    """
    Interactive shell for the user to set missingness.

    Can only be used for Missingness Completely At Random.
    """

    intro = "Interactive missingness configuration. Type ? for help.\n"
    doc_leader = """Use commands 'sampled' and 'none' to choose the missingness style for
the current table. Use commands 'next' and 'previous' to change the
current table. Use 'tables' to list the tables and 'count' to show
how many NULLs exist in each column. Use 'peek' or 'select' to see
data from the database. Use 'quit' to exit this tool."""
    prompt = "(missingness) "
    file = None
    PATTERN_RE = re.compile(r'SRC_STATS\["([^"]*)"\]')

    def find_missingness_query(
        self, missingness_generator: Mapping
    ) -> tuple[str, str] | None:
        """Find query and comment from src-stats for the passed missingness generator."""
        kwargs = missingness_generator.get("kwargs", {})
        patterns = kwargs.get("patterns", "")
        pattern_match = self.PATTERN_RE.match(patterns)
        if pattern_match:
            key = pattern_match.group(1)
            for src_stat in self.config["src-stats"]:
                if src_stat.get("name") == key:
                    query = src_stat.get("query", None)
                    if type(query) is not str:
                        return None
                    return (query, src_stat.get("comment", ""))
        return None

    def make_table_entry(
        self, name: str, table: Mapping
    ) -> MissingnessCmdTableEntry | None:
        """
        Make a table entry for a particular table.

        :param name: The name of the table to make an entry for.
        :param table: The part of ``config.yaml`` relating to this table.
        :return: The newly-constructed table entry.
        """
        if table.get("ignore", False):
            return None
        if table.get("vocabulary_table", False):
            return None
        if table.get("num_rows_per_pass", 1) == 0:
            return None
        mgs = table.get("missingness_generators", [])
        old = None
        nonnull_columns = self.get_nonnull_columns(name)
        if not nonnull_columns:
            return None
        if not mgs:
            old = MissingnessType(
                name="none",
                query="",
                comment="",
                columns=[],
            )
        elif len(mgs) == 1:
            mg = mgs[0]
            mg_name = mg.get("name", None)
            if type(mg_name) is str:
                query_comment = self.find_missingness_query(mg)
                if query_comment is not None:
                    (query, comment) = query_comment
                    old = MissingnessType(
                        name=mg_name,
                        query=query,
                        comment=comment,
                        columns=mg.get("columns_assigned", []),
                    )
        if old is None:
            return None
        return MissingnessCmdTableEntry(
            name=name,
            old_type=old,
            new_type=old,
        )

    def __init__(
        self,
        src_dsn: str,
        src_schema: str | None,
        metadata: MetaData,
        config: MutableMapping,
    ):
        """
        Initialise a MissingnessCmd.

        :param src_dsn: connection string for the source database.
        :param src_schema: schema name for the source database.
        :param metadata: SQLAlchemy metadata for the source database.
        :param config: Configuration from the ``config.yaml`` file.
        """
        super().__init__(src_dsn, src_schema, metadata, config)
        self.set_prompt()

    @property
    def table_entries(self) -> list[MissingnessCmdTableEntry]:
        """Get the table entries list."""
        return cast(list[MissingnessCmdTableEntry], self._table_entries)

    def find_entry_by_table_name(
        self, table_name: str
    ) -> MissingnessCmdTableEntry | None:
        """Find the table entry given the table name."""
        entry = super().find_entry_by_table_name(table_name)
        if entry is None:
            return None
        return cast(MissingnessCmdTableEntry, entry)

    def set_prompt(self) -> None:
        """Set the prompt according to the current table and missingness."""
        if self.table_index < len(self.table_entries):
            entry: MissingnessCmdTableEntry = self.table_entries[self.table_index]
            nt = entry.new_type
            if nt is None:
                self.prompt = "(missingness for {0}) ".format(entry.name)
            else:
                self.prompt = "(missingness for {0}: {1}) ".format(entry.name, nt.name)
        else:
            self.prompt = "(missingness) "

    def set_type(self, t_type: MissingnessType) -> None:
        """Set the missingness of the current table."""
        if self.table_index < len(self.table_entries):
            entry = self.table_entries[self.table_index]
            entry.new_type = t_type

    def _copy_entries(self) -> None:
        """Set the new missingness into the configuration."""
        src_stats = self._remove_prefix_src_stats("missing_auto__")
        for entry in self.table_entries:
            table = self.get_table_config(entry.name)
            if entry.new_type is None or entry.new_type.name == "none":
                table.pop("missingness_generators", None)
            else:
                src_stat_key = "missing_auto__{0}__0".format(entry.name)
                table["missingness_generators"] = [
                    {
                        "name": entry.new_type.name,
                        "kwargs": {
                            "patterns": 'SRC_STATS["{0}"]["results"]'.format(
                                src_stat_key
                            )
                        },
                        "columns": entry.new_type.columns,
                    }
                ]
                src_stats.append(
                    {
                        "name": src_stat_key,
                        "query": entry.new_type.query,
                        "comments": []
                        if entry.new_type.comment is None
                        else [entry.new_type.comment],
                    }
                )
            self.set_table_config(entry.name, table)

    def do_quit(self, _arg: str) -> bool:
        """Check the updates, save them if desired and quit the configurer."""
        count = 0
        for entry in self.table_entries:
            if entry.old_type != entry.new_type:
                count += 1
                if entry.old_type is None:
                    self.print(
                        "Putting generator {0} on table {1}",
                        entry.name,
                        entry.new_type.name,
                    )
                elif entry.new_type is None:
                    self.print(
                        "Deleting generator {1} from table {0}",
                        entry.name,
                        entry.old_type.name,
                    )
                else:
                    self.print(
                        "Changing {0} from {1} to {2}",
                        entry.name,
                        entry.old_type.name,
                        entry.new_type.name,
                    )
        if count == 0:
            self.print("You have made no changes.")
        reply = self.ask_save()
        if reply == "yes":
            self._copy_entries()
            return True
        if reply == "no":
            return True
        return False

    def do_tables(self, _arg: str) -> None:
        """List the tables with their types."""
        for entry in self.table_entries:
            old = "-" if entry.old_type is None else entry.old_type.name
            new = "-" if entry.new_type is None else entry.new_type.name
            desc = new if old == new else "{0}->{1}".format(old, new)
            self.print("{0} {1}", entry.name, desc)

    def do_next(self, arg: str) -> None:
        """
        Go to the next table, or a specified table.

        'next' = go to the next table, 'next tablename' = go to table 'tablename'
        """
        if arg:
            # Find the index of the table called _arg, if any
            index = next(
                (i for i, entry in enumerate(self.table_entries) if entry.name == arg),
                None,
            )
            if index is None:
                self.print(self.ERROR_NO_SUCH_TABLE, arg)
                return
            self.set_table_index(index)
            return
        self.next_table(self.INFO_NO_MORE_TABLES)

    def complete_next(
        self, text: str, _line: str, _begidx: int, _endidx: int
    ) -> list[str]:
        """Get completions for tables and columns."""
        return [
            entry.name for entry in self.table_entries if entry.name.startswith(text)
        ]

    def do_previous(self, _arg: str) -> None:
        """Go to the previous table."""
        if not self.set_table_index(self.table_index - 1):
            self.print(self.ERROR_ALREADY_AT_START)

    def _set_type(self, name: str, query: str, comment: str) -> None:
        """Set the current table entry's query."""
        if len(self.table_entries) <= self.table_index:
            return
        entry: MissingnessCmdTableEntry = self.table_entries[self.table_index]
        entry.new_type = MissingnessType(
            name=name,
            query=query,
            comment=comment,
            columns=self.get_nonnull_columns(entry.name),
        )

    def _set_none(self) -> None:
        """Set the current table to have no missingness applied."""
        if len(self.table_entries) <= self.table_index:
            return
        self.table_entries[self.table_index].new_type = None

    def do_sampled(self, arg: str) -> None:
        """
        Set the current table missingness as 'sampled', and go to the next table.

        'sampled 3000' means sample 3000 rows at random and choose the
        missingness to be the same as one of those 3000 at random.
        'sampled' means the same, but with a default number of rows sampled (1000).
        """
        if len(self.table_entries) <= self.table_index:
            self.print("Error! not on a table")
            return
        entry = self.table_entries[self.table_index]
        if arg == "":
            count = 1000
        elif arg.isdecimal():
            count = int(arg)
        else:
            self.print(
                "Error: sampled can be used alone or with an integer argument. {0} is not permitted",
                arg,
            )
            return
        self._set_type(
            MissingnessType.SAMPLED,
            MissingnessType.sampled_query(
                entry.name,
                count,
                self.get_nonnull_columns(entry.name),
            ),
            f"The missingness patterns and how often they appear in a sample of {count} from table {entry.name}",
        )
        self.print("Table {} set to sampled missingness", self.table_name())
        self.next_table()

    def do_none(self, _arg: str) -> None:
        """Set the current table to have no missingness, and go to the next table."""
        self._set_none()
        self.print("Table {} set to have no missingness", self.table_name())
        self.next_table()


def update_missingness(
    src_dsn: str,
    src_schema: str | None,
    metadata: MetaData,
    config: MutableMapping[str, Any],
) -> Mapping[str, Any]:
    """
    Ask the user to update the missingness information in ``config.yaml``.

    :param src_dsn: The connection string for the source database.
    :param src_schema: The name of the source database schema (or None
    for the default).
    :param metadata: The SQLAlchemy metadata object from ``orm.yaml``.
    :param config: The starting configuration,
    :return: The updated configuration.
    """
    with MissingnessCmd(src_dsn, src_schema, metadata, config) as mc:
        mc.cmdloop()
        return mc.config


@dataclass
class GeneratorInfo:
    """A generator and the columns it assigns to."""

    columns: list[str]
    gen: Generator | None


@dataclass
class GeneratorCmdTableEntry(TableEntry):
    """
    List of generators set for a table.

    Includes the original setting and the currently configured
    generators.
    """

    old_generators: list[GeneratorInfo]
    new_generators: list[GeneratorInfo]


class GeneratorCmd(DbCmd):
    """Interactive command shell for setting generators."""

    intro = "Interactive generator configuration. Type ? for help.\n"
    doc_leader = """Use command 'propose' for a list of generators applicable to the
current column, then command 'compare' to see how these perform
against the source data, then command 'set' to choose your favourite.
Use 'unset' to remove the column's generator. Use commands 'next' and
'previous' to change which column we are examining. Use 'info'
for useful information about the current column. Use 'tables' and
'list' to see available tables and columns. Use 'columns' to see
information about the columns in the current table. Use 'peek',
'count' or 'select' to fetch data from the source database. Use
'quit' to exit this program."""
    prompt = "(generatorconf) "
    file = None

    PROPOSE_SOURCE_SAMPLE_TEXT = "Sample of actual source data: {0}..."
    PROPOSE_SOURCE_EMPTY_TEXT = "Source database has no data in this column."
    PROPOSE_GENERATOR_SAMPLE_TEXT = "{index}. {name}: {fit} {sample} ..."
    PRIMARY_PRIVATE_TEXT = "Primary Private"
    SECONDARY_PRIVATE_TEXT = "Secondary Private on columns {0}"
    NOT_PRIVATE_TEXT = "Not private"
    ERROR_NO_SUCH_TABLE = "No such (non-vocabulary, non-ignored) table name {0}"
    ERROR_NO_SUCH_COLUMN = "No such column {0} in this table"
    ERROR_COLUMN_ALREADY_MERGED = "Column {0} is already merged"
    ERROR_COLUMN_ALREADY_UNMERGED = "Column {0} is not merged"
    ERROR_CANNOT_UNMERGE_ALL = "You cannot unmerge all the generator's columns"
    PROPOSE_NOTHING = "No proposed generators, sorry."

    SRC_STAT_RE = re.compile(
        r'\bSRC_STATS\["([^"]+)"\](\["results"\]\[0\]\["([^"]+)"\])?'
    )

    def make_table_entry(
        self, table_name: str, table: Mapping
    ) -> GeneratorCmdTableEntry | None:
        """
        Make a table entry.

        :param table_name: The name of the table.
        :param table: The portion of the ``config.yaml`` file describing this table.
        :return: The newly constructed table entry, or None if this table is to be ignored.
        """
        if table.get("ignore", False):
            return None
        if table.get("vocabulary_table", False):
            return None
        if table.get("num_rows_per_pass", 1) == 0:
            return None
        metadata_table = self.metadata.tables[table_name]
        columns = [str(colname) for colname in metadata_table.columns.keys()]
        column_set = frozenset(columns)
        columns_assigned_so_far: set[str] = set()

        new_generator_infos: list[GeneratorInfo] = []
        old_generator_infos: list[GeneratorInfo] = []
        for rg in table.get("row_generators", []):
            gen_name = rg.get("name", None)
            if gen_name:
                ca = rg.get("columns_assigned", [])
                collist: list[str] = (
                    [ca] if isinstance(ca, str) else [str(c) for c in ca]
                )
                colset: set[str] = set(collist)
                for unknown in colset - column_set:
                    logger.warning(
                        "table '%s' has '%s' assigned to column '%s' which is not in this table",
                        table_name,
                        gen_name,
                        unknown,
                    )
                for mult in columns_assigned_so_far & colset:
                    logger.warning(
                        "table '%s' has column '%s' assigned to multiple times",
                        table_name,
                        mult,
                    )
                actual_collist = [c for c in collist if c in columns]
                if actual_collist:
                    gen = PredefinedGenerator(table_name, rg, self.config)
                    new_generator_infos.append(
                        GeneratorInfo(
                            columns=actual_collist.copy(),
                            gen=gen,
                        )
                    )
                    old_generator_infos.append(
                        GeneratorInfo(
                            columns=actual_collist.copy(),
                            gen=gen,
                        )
                    )
                    columns_assigned_so_far |= colset
        for colname in columns:
            if colname not in columns_assigned_so_far:
                new_generator_infos.append(
                    GeneratorInfo(
                        columns=[colname],
                        gen=None,
                    )
                )
        if len(new_generator_infos) == 0:
            return None
        return GeneratorCmdTableEntry(
            name=table_name,
            old_generators=old_generator_infos,
            new_generators=new_generator_infos,
        )

    def __init__(
        self,
        src_dsn: str,
        src_schema: str | None,
        metadata: MetaData,
        config: MutableMapping[str, Any],
    ) -> None:
        """
        Initialise a ``GeneratorCmd``.

        :param src_dsn: connection address for source database
        :param src_schema: database schema name
        :param metadata: SQLAlchemy metadata for the source database
        :param config: Configuration loaded from ``config.yaml``
        """
        super().__init__(src_dsn, src_schema, metadata, config)
        self.generator_index = 0
        self.generators_valid_columns: Optional[tuple[int, list[str]]] = None
        self.set_prompt()

    @property
    def table_entries(self) -> list[GeneratorCmdTableEntry]:
        """Get the talbe entries, cast to ``GeneratorCmdTableEntry``."""
        return cast(list[GeneratorCmdTableEntry], self._table_entries)

    def find_entry_by_table_name(
        self, table_name: str
    ) -> GeneratorCmdTableEntry | None:
        """
        Find the table entry by name.

        :param table_name: The name of the table to find.
        :return: The table entry, or None if no such table name exists.
        """
        entry = super().find_entry_by_table_name(table_name)
        if entry is None:
            return None
        return cast(GeneratorCmdTableEntry, entry)

    def set_table_index(self, index: int) -> bool:
        """
        Move to a new table.

        :param index: table index to move to.
        """
        ret = super().set_table_index(index)
        if ret:
            self.generator_index = 0
            self.set_prompt()
        return ret

    def previous_table(self) -> bool:
        """
        Move to the table before the current one.

        :return: True if there is a previous table to go to.
        """
        ret = self.set_table_index(self.table_index - 1)
        if ret:
            table = self.get_table()
            if table is None:
                self.print(
                    "Internal error! table {0} does not have any generators!",
                    self.table_index,
                )
                return False
            self.generator_index = len(table.new_generators) - 1
        else:
            self.print(self.ERROR_ALREADY_AT_START)
        return ret

    def get_table(self) -> GeneratorCmdTableEntry | None:
        """Get the current table entry."""
        if self.table_index < len(self.table_entries):
            return self.table_entries[self.table_index]
        return None

    def get_table_and_generator(self) -> tuple[str | None, GeneratorInfo | None]:
        """Get a pair; the table name then the generator information."""
        if self.table_index < len(self.table_entries):
            entry = self.table_entries[self.table_index]
            if self.generator_index < len(entry.new_generators):
                return (entry.name, entry.new_generators[self.generator_index])
            return (entry.name, None)
        return (None, None)

    def get_column_names(self) -> list[str]:
        """Get the (unqualified) names for all the current columns."""
        (_, generator_info) = self.get_table_and_generator()
        return generator_info.columns if generator_info else []

    def column_metadata(self) -> list[Column]:
        """Get the metadata for all the current columns."""
        table = self.table_metadata()
        if table is None:
            return []
        return [table.columns[name] for name in self.get_column_names()]

    def set_prompt(self) -> None:
        """Set the prompt according to the current table, column and generator."""
        (table_name, gen_info) = self.get_table_and_generator()
        if table_name is None:
            self.prompt = "(generators) "
            return
        if gen_info is None:
            self.prompt = "({table}) ".format(table=table_name)
            return
        table = self.table_metadata()
        columns = [
            c + "[pk]" if table.columns[c].primary_key else c for c in gen_info.columns
        ]
        gen = f" ({gen_info.gen.name()})" if gen_info.gen else ""
        self.prompt = f"({table_name}.{','.join(columns)}{gen}) "

    def _remove_auto_src_stats(self) -> list[dict[str, Any]]:
        """
        Remove all automatic source stats.

        We assume every source stats query whose name begins with ``auto__`
        :return: The new ``src_stats`` configuration.
        """
        return self._remove_prefix_src_stats("auto__")

    def _copy_entries(self) -> None:
        """Set generator and query information in the configuration."""
        src_stats = self._remove_auto_src_stats()
        for entry in self.table_entries:
            rgs = []
            new_gens: list[Generator] = []
            for generator in entry.new_generators:
                if generator.gen is not None:
                    new_gens.append(generator.gen)
                    cqs = generator.gen.custom_queries()
                    for cq_key, cq in cqs.items():
                        src_stats.append(
                            {
                                "name": cq_key,
                                "query": cq["query"],
                                "comments": [cq["comment"]]
                                if "comment" in cq and cq["comment"]
                                else [],
                            }
                        )
                    rg: dict[str, Any] = {
                        "name": generator.gen.function_name(),
                        "columns_assigned": generator.columns,
                    }
                    kwn = generator.gen.nominal_kwargs()
                    if kwn:
                        rg["kwargs"] = kwn
                    rgs.append(rg)
            aq = self._get_aggregate_query(new_gens, entry.name)
            if aq:
                src_stats.append(
                    {
                        "name": f"auto__{entry.name}",
                        "query": aq,
                        "comments": [
                            q["comment"]
                            for gen in new_gens
                            for q in gen.select_aggregate_clauses().values()
                            if "comment" in q and q["comment"] is not None
                        ],
                    }
                )
            table_config = self.get_table_config(entry.name)
            if rgs:
                table_config["row_generators"] = rgs
            elif "row_generators" in table_config:
                del table_config["row_generators"]
            self.set_table_config(entry.name, table_config)
        self.config["src-stats"] = src_stats

    def _find_old_generator(
        self, entry: GeneratorCmdTableEntry, columns: Iterable[str]
    ) -> Generator | None:
        """Find any generator that previously assigned to these exact same columns."""
        fc = frozenset(columns)
        for gen in entry.old_generators:
            if frozenset(gen.columns) == fc:
                return gen.gen
        return None

    def do_quit(self, arg: str) -> bool:
        """Check the updates, save them if desired and quit the configurer."""
        count = 0
        for entry in self.table_entries:
            header_shown = False
            g_entry = cast(GeneratorCmdTableEntry, entry)
            for gen in g_entry.new_generators:
                old_gen = self._find_old_generator(g_entry, gen.columns)
                new_gen = None if gen is None else gen.gen
                if old_gen != new_gen:
                    if not header_shown:
                        header_shown = True
                        self.print("Table {0}:", entry.name)
                    count += 1
                    self.print(
                        "...changing {0} from {1} to {2}",
                        ", ".join(gen.columns),
                        old_gen.name() if old_gen else "nothing",
                        gen.gen.name() if gen.gen else "nothing",
                    )
        if count == 0:
            self.print("You have made no changes.")
        if arg in {"yes", "no"}:
            reply = arg
        else:
            reply = self.ask_save()
        if reply == "yes":
            self._copy_entries()
            return True
        if reply == "no":
            return True
        return False

    def do_tables(self, arg: str) -> None:
        """List the tables."""
        for t_entry in self.table_entries:
            entry = cast(GeneratorCmdTableEntry, t_entry)
            gen_count = len(entry.new_generators)
            how_many = "one generator" if gen_count == 1 else f"{gen_count} generators"
            self.print("{0} ({1})", entry.name, how_many)

    def do_list(self, arg: str) -> None:
        """List the generators in the current table."""
        if len(self.table_entries) <= self.table_index:
            self.print("Error: no table {0}", self.table_index)
            return
        g_entry = cast(GeneratorCmdTableEntry, self.table_entries[self.table_index])
        table = self.table_metadata()
        for gen in g_entry.new_generators:
            old_gen = self._find_old_generator(g_entry, gen.columns)
            old = "" if old_gen is None else old_gen.name()
            if old_gen == gen.gen:
                becomes = ""
                if old == "":
                    old = "(not set)"
            elif gen.gen is None:
                becomes = "(delete)"
            else:
                becomes = f"->{gen.gen.name()}"
            primary = ""
            if len(gen.columns) == 1 and table.columns[gen.columns[0]].primary_key:
                primary = "[primary-key]"
            self.print("{0}{1}{2} {3}", old, becomes, primary, gen.columns)

    def do_columns(self, _arg: str) -> None:
        """Report the column names and metadata."""
        self.report_columns()

    def do_info(self, _arg: str) -> None:
        """Show information about the current column."""
        for cm in self.column_metadata():
            self.print(
                "Column {0} in table {1} has type {2} ({3}).",
                cm.name,
                cm.table.name,
                str(cm.type),
                "nullable" if cm.nullable else "not nullable",
            )
            if cm.primary_key:
                self.print(
                    "It is a primary key, which usually does not need a generator (it will auto-increment)"
                )
            if cm.foreign_keys:
                fk_names = [fk_column_name(fk) for fk in cm.foreign_keys]
                self.print(
                    "It is a foreign key referencing column {0}", ", ".join(fk_names)
                )
                if len(fk_names) == 1 and not cm.primary_key:
                    self.print(
                        "You do not need a generator if you just want a uniform choice over the referenced table's rows"
                    )

    def _get_table_index(self, table_name: str) -> int | None:
        """Get the index of the named table in the table entries list."""
        for n, entry in enumerate(self.table_entries):
            if entry.name == table_name:
                return n
        return None

    def _get_generator_index(self, table_index: int, column_name: str) -> int | None:
        """
        Get the index number of a column within the list of generators in this table.

        :param table_index: The index of the table in which to search.
        :param column_name: The name of the column to search for.
        :return: The index in the ``new_generators`` attribute of the table entry
        containing the specified column, or None if this does not exist.
        """
        entry = self.table_entries[table_index]
        for n, gen in enumerate(entry.new_generators):
            if column_name in gen.columns:
                return n
        return None

    def go_to(self, target: str) -> bool:
        """
        Go to a particular column.

        :return: True on success.
        """
        parts = target.split(".", 1)
        table_index = self._get_table_index(parts[0])
        if table_index is None:
            if len(parts) == 1:
                gen_index = self._get_generator_index(self.table_index, parts[0])
                if gen_index is not None:
                    self.generator_index = gen_index
                    self.set_prompt()
                    return True
            self.print(self.ERROR_NO_SUCH_TABLE_OR_COLUMN, parts[0])
            return False
        gen_index = None
        if 1 < len(parts) and parts[1]:
            gen_index = self._get_generator_index(table_index, parts[1])
            if gen_index is None:
                self.print("we cannot set the generator for column {0}", parts[1])
                return False
        self.set_table_index(table_index)
        if gen_index is not None:
            self.generator_index = gen_index
            self.set_prompt()
        return True

    def do_next(self, arg: str) -> None:
        """
        Go to the next generator. or a specified generator.

        Go to a named table: 'next tablename',
        go to a column: 'next tablename.columnname',
        or go to a column within this table: 'next columnname'.
        """
        if arg:
            self.go_to(arg)
        else:
            self._go_next()

    def do_n(self, arg: str) -> None:
        """Go to the next generator, or a specified generator."""
        self.do_next(arg)

    def complete_n(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        """Complete the ``n`` command's arguments."""
        return self.complete_next(text, line, begidx, endidx)

    def _go_next(self) -> None:
        """Go to the next column."""
        table = self.get_table()
        if table is None:
            self.print("No more tables")
            return
        next_gi = self.generator_index + 1
        if next_gi == len(table.new_generators):
            self.next_table(self.INFO_NO_MORE_TABLES)
            return
        self.generator_index = next_gi
        self.set_prompt()

    def complete_next(
        self, text: str, _line: str, _begidx: int, _endidx: int
    ) -> list[str]:
        """Completions for the arguments of the ``next`` command."""
        parts = text.split(".", 1)
        first_part = parts[0]
        if 1 < len(parts):
            column_name = parts[1]
            table_index = self._get_table_index(first_part)
            if table_index is None:
                return []
            table_entry = self.table_entries[table_index]
            return [
                f"{first_part}.{column}"
                for gen in table_entry.new_generators
                for column in gen.columns
                if column.startswith(column_name)
            ]
        table_names = [
            entry.name
            for entry in self.table_entries
            if entry.name.startswith(first_part)
        ]
        if first_part in table_names:
            table_names.append(f"{first_part}.")
        current_table = self.get_table()
        if current_table:
            column_names = [
                col
                for gen in current_table.new_generators
                for col in gen.columns
                if col.startswith(first_part)
            ]
        else:
            column_names = []
        return table_names + column_names

    def do_previous(self, _arg: str) -> None:
        """Go to the previous generator."""
        if self.generator_index == 0:
            self.previous_table()
        else:
            self.generator_index -= 1
        self.set_prompt()

    def do_b(self, arg: str) -> None:
        """Synonym for previous."""
        self.do_previous(arg)

    def _generators_valid(self) -> bool:
        """Test if ``self.generators`` is still correct for the current columns."""
        return self.generators_valid_columns == (
            self.table_index,
            self.get_column_names(),
        )

    def _get_generator_proposals(self) -> list[Generator]:
        """Get a list of acceptable generators, sorted by decreasing fit to the actual data."""
        if not self._generators_valid():
            self.generators = None
        if self.generators is None:
            columns = self.column_metadata()
            gens = everything_factory().get_generators(columns, self.sync_engine)
            sorted_gens = sorted(gens, key=lambda g: g.fit(9999))
            self.generators = sorted_gens
            self.generators_valid_columns = (
                self.table_index,
                self.get_column_names().copy(),
            )
        return self.generators

    def _print_privacy(self) -> None:
        """Print the privacy status of the current table."""
        table = self.table_metadata()
        if table is None:
            return
        if table_is_private(self.config, table.name):
            self.print(self.PRIMARY_PRIVATE_TEXT)
            return
        pfks = primary_private_fks(self.config, table)
        if not pfks:
            self.print(self.NOT_PRIVATE_TEXT)
            return
        self.print(self.SECONDARY_PRIVATE_TEXT, pfks)

    def do_compare(self, arg: str) -> None:
        """
        Compare the real data with some generators.

        'compare': just look at some source data from this column.
        'compare 5 6 10': compare a sample of the source data with a sample
        from generators 5, 6 and 10. You can find out which numbers
        correspond to which generators using the 'propose' command.
        """
        self._print_privacy()
        args = arg.split()
        limit = 20
        comparison = {
            "source": [
                x[0] if len(x) == 1 else ", ".join(x)
                for x in self._get_column_data(limit, to_str=str)
            ]
        }
        gens: list[Generator] = self._get_generator_proposals()
        table_name = self.table_name()
        for argument in args:
            if argument.isdigit():
                n = int(argument)
                if 0 < n and n <= len(gens):
                    gen = gens[n - 1]
                    comparison[f"{n}. {gen.name()}"] = gen.generate_data(limit)
                    self._print_values_queried(table_name, n, gen)
        self.print_table_by_columns(comparison)

    def do_c(self, arg: str) -> None:
        """Synonym for compare."""
        self.do_compare(arg)

    def _print_values_queried(self, table_name: str, n: int, gen: Generator) -> None:
        """
        Print the values queried from the database for this generator.

        :param table_name: The name of the table the generator applies to.
        :param n: A number to print at the start of the output.
        :param gen: The generator to report.
        """
        if not gen.select_aggregate_clauses() and not gen.custom_queries():
            self.print(
                "{0}. {1} requires no data from the source database.",
                n,
                gen.name(),
            )
        else:
            self.print(
                "{0}. {1} requires the following data from the source database:",
                n,
                gen.name(),
            )
            self._print_select_aggregate_query(table_name, gen)
            self._print_custom_queries(gen)

    def _print_custom_queries(self, gen: Generator) -> None:
        """
        Print all the custom queries and all the values they get in this case.

        :param gen: The generator to print the custom queries for.
        """
        cqs = gen.custom_queries()
        if not cqs:
            return
        cq_key2args: dict[str, Any] = {}
        nominal = gen.nominal_kwargs()
        actual = gen.actual_kwargs()
        self._get_custom_queries_from(
            cq_key2args,
            nominal,
            actual,
        )
        for cq_key, cq in cqs.items():
            self.print(
                "{0}; providing the following values: {1}",
                cq["query"],
                cq_key2args[cq_key],
            )

    def _get_custom_queries_from(
        self, out: dict[str, Any], nominal: Any, actual: Any
    ) -> None:
        if type(nominal) is str:
            src_stat_groups = self.SRC_STAT_RE.search(nominal)
            # Do we have a SRC_STAT reference?
            if src_stat_groups:
                # Get its name
                cq_key = src_stat_groups.group(1)
                # Are we pulling a specific part of this result?
                sub = src_stat_groups.group(3)
                if sub:
                    actual = {sub: actual}
                else:
                    out[cq_key] = actual
        elif type(nominal) is list and type(actual) is list:
            for i in range(min(len(nominal), len(actual))):
                self._get_custom_queries_from(out, nominal[i], actual[i])
        elif type(nominal) is dict and type(actual) is dict:
            for k, v in nominal.items():
                if k in actual:
                    self._get_custom_queries_from(out, v, actual[k])

    def _get_aggregate_query(
        self, gens: list[Generator], table_name: str
    ) -> str | None:
        clauses = [
            f'{q["clause"]} AS {n}'
            for gen in gens
            for n, q in or_default(gen.select_aggregate_clauses(), {}).items()
        ]
        if not clauses:
            return None
        return f"SELECT {', '.join(clauses)} FROM {table_name}"

    def _print_select_aggregate_query(self, table_name: str, gen: Generator) -> None:
        """
        Print the select aggregate query and all the values it gets in this case.

        This is not the entire query that will be executed, but only the part of it
        that is required by a certain generator.
        :param table_name: The table name.
        :param gen: The generator to limit the aggregate query to.
        """
        sacs = gen.select_aggregate_clauses()
        if not sacs:
            return
        kwa = gen.actual_kwargs()
        vals = []
        src_stat2kwarg = {v: k for k, v in gen.nominal_kwargs().items()}
        for n in sacs.keys():
            src_stat = f'SRC_STATS["auto__{table_name}"]["results"][0]["{n}"]'
            if src_stat in src_stat2kwarg:
                ak = src_stat2kwarg[src_stat]
                if ak in kwa:
                    vals.append(kwa[ak])
                else:
                    logger.warning(
                        "actual_kwargs for %s does not report %s", gen.name(), ak
                    )
            else:
                logger.warning(
                    'nominal_kwargs for %s does not have a value SRC_STATS["auto__%s"]["results"][0]["%s"]',
                    gen.name(),
                    table_name,
                    n,
                )
        select_q = self._get_aggregate_query([gen], table_name)
        self.print("{0}; providing the following values: {1}", select_q, vals)

    def _get_column_data(
        self, count: int, to_str: Callable[[Any], str] = repr
    ) -> list[list[str]]:
        columns = self.get_column_names()
        columns_string = ", ".join(columns)
        pred = " AND ".join(f"{column} IS NOT NULL" for column in columns)
        with self.sync_engine.connect() as connection:
            result = connection.execute(
                text(
                    f"SELECT {columns_string} FROM {self.table_name()} WHERE {pred} ORDER BY RANDOM() LIMIT {count}"
                )
            )
            return [[to_str(x) for x in xs] for xs in result.all()]

    def do_propose(self, _arg: str) -> None:
        """
        Display a list of possible generators for this column.

        They will be listed in order of fit, the most likely matches first.
        The results can be compared (against a sample of the real data in
        the column and against each other) with the 'compare' command.
        """
        limit = 5
        gens = self._get_generator_proposals()
        sample = self._get_column_data(limit)
        if sample:
            rep = [x[0] if len(x) == 1 else ",".join(x) for x in sample]
            self.print(self.PROPOSE_SOURCE_SAMPLE_TEXT, "; ".join(rep))
        else:
            self.print(self.PROPOSE_SOURCE_EMPTY_TEXT)
        if not gens:
            self.print(self.PROPOSE_NOTHING)
        for index, gen in enumerate(gens):
            fit = gen.fit(-1)
            if fit == -1:
                fit_s = "(no fit)"
            elif fit < 100:
                fit_s = f"(fit: {fit:.3g})"
            else:
                fit_s = f"(fit: {fit:.0f})"
            self.print(
                self.PROPOSE_GENERATOR_SAMPLE_TEXT,
                index=index + 1,
                name=gen.name(),
                fit=fit_s,
                sample="; ".join(map(repr, gen.generate_data(limit))),
            )

    def do_p(self, arg: str) -> None:
        """Synonym for propose."""
        self.do_propose(arg)

    def get_proposed_generator_by_name(self, gen_name: str) -> Generator | None:
        """Find a generator by name from the list of proposals."""
        for gen in self._get_generator_proposals():
            if gen.name() == gen_name:
                return gen
        return None

    def do_set(self, arg: str) -> None:
        """Set one of the proposals as a generator."""
        if arg.isdigit() and not self._generators_valid():
            self.print("Please run 'propose' before 'set <number>'")
            return
        gens = self._get_generator_proposals()
        new_gen: Generator | None
        if arg.isdigit():
            index = int(arg)
            if index < 1:
                self.print("set's integer argument must be at least 1")
                return
            if len(gens) < index:
                self.print(
                    "There are currently only {0} generators proposed, please select one of them.",
                    len(gens),
                )
                return
            new_gen = gens[index - 1]
        else:
            new_gen = self.get_proposed_generator_by_name(arg)
            if new_gen is None:
                self.print("'{0}' is not an appropriate generator for this column", arg)
                return
        self.set_generator(new_gen)
        self._go_next()

    def set_generator(self, gen: Generator | None) -> None:
        """Set the current column's generator."""
        (table, gen_info) = self.get_table_and_generator()
        if table is None:
            self.print("Error: no table")
            return
        if gen_info is None:
            self.print("Error: no column")
            return
        gen_info.gen = gen

    def do_s(self, arg: str) -> None:
        """Synonym for set."""
        self.do_set(arg)

    def do_unset(self, _arg: str) -> None:
        """Remove any generator set for this column."""
        self.set_generator(None)
        self._go_next()

    def do_merge(self, arg: str) -> None:
        """
        Add this column(s) to the specified column(s).

        After this, one generator will cover them all.
        """
        cols = arg.split()
        if not cols:
            self.print("Error: merge requires a column argument")
        table_entry: GeneratorCmdTableEntry | None = self.get_table()
        if table_entry is None:
            self.print(self.ERROR_NO_SUCH_TABLE)
            return
        cols_available = functools.reduce(
            lambda x, y: x | y,
            [frozenset(gen.columns) for gen in table_entry.new_generators],
        )
        cols_to_merge = frozenset(cols)
        unknown_cols = cols_to_merge - cols_available
        if unknown_cols:
            for uc in unknown_cols:
                self.print(self.ERROR_NO_SUCH_COLUMN, uc)
            return
        gen_info = table_entry.new_generators[self.generator_index]
        current_columns = frozenset(gen_info.columns)
        stated_current_columns = cols_to_merge & current_columns
        if stated_current_columns:
            for c in stated_current_columns:
                self.print(self.ERROR_COLUMN_ALREADY_MERGED, c)
            return
        # Remove cols_to_merge from each generator
        new_new_generators: list[GeneratorInfo] = []
        for gen in table_entry.new_generators:
            if gen is gen_info:
                # Add columns to this generator
                self.generator_index = len(new_new_generators)
                new_new_generators.append(
                    GeneratorInfo(
                        columns=gen.columns + cols,
                        gen=None,
                    )
                )
            else:
                # Remove columns if applicable
                new_columns = [c for c in gen.columns if c not in cols_to_merge]
                is_changed = len(new_columns) != len(gen.columns)
                if new_columns:
                    # We have not removed this generator completely
                    new_new_generators.append(
                        GeneratorInfo(
                            columns=new_columns,
                            gen=None if is_changed else gen.gen,
                        )
                    )
        table_entry.new_generators = new_new_generators
        self.set_prompt()

    def complete_merge(
        self, text: str, _line: str, _begidx: int, _endidx: int
    ) -> list[str]:
        """Complete column names."""
        last_arg = text.split()[-1]
        table_entry: GeneratorCmdTableEntry | None = self.get_table()
        if table_entry is None:
            return []
        return [
            column
            for i, gen in enumerate(table_entry.new_generators)
            if i != self.generator_index
            for column in gen.columns
            if column.startswith(last_arg)
        ]

    def do_unmerge(self, arg: str) -> None:
        """Remove this column(s) from this generator, make them a separate generator."""
        cols = arg.split()
        if not cols:
            self.print("Error: merge requires a column argument")
        table_entry: GeneratorCmdTableEntry | None = self.get_table()
        if table_entry is None:
            self.print(self.ERROR_NO_SUCH_TABLE)
            return
        gen_info = table_entry.new_generators[self.generator_index]
        current_columns = frozenset(gen_info.columns)
        cols_to_unmerge = frozenset(cols)
        unknown_cols = cols_to_unmerge - current_columns
        if unknown_cols:
            for uc in unknown_cols:
                self.print(self.ERROR_NO_SUCH_COLUMN, uc)
            return
        stated_unmerged_columns = cols_to_unmerge - current_columns
        if stated_unmerged_columns:
            for c in stated_unmerged_columns:
                self.print(self.ERROR_COLUMN_ALREADY_UNMERGED, c)
            return
        if cols_to_unmerge == current_columns:
            self.print(self.ERROR_CANNOT_UNMERGE_ALL)
            return
        # Remove unmerged columns
        for um in cols_to_unmerge:
            gen_info.columns.remove(um)
        # The existing generator will not work
        gen_info.gen = None
        # And put them into a new (empty) generator
        table_entry.new_generators.insert(
            self.generator_index + 1,
            GeneratorInfo(
                columns=cols,
                gen=None,
            ),
        )
        self.set_prompt()

    def complete_unmerge(
        self, text: str, _line: str, _begidx: int, _endidx: int
    ) -> list[str]:
        """Complete column names to unmerge."""
        last_arg = text.split()[-1]
        table_entry: GeneratorCmdTableEntry | None = self.get_table()
        if table_entry is None:
            return []
        return [
            column
            for column in table_entry.new_generators[self.generator_index].columns
            if column.startswith(last_arg)
        ]


def update_config_generators(
    src_dsn: str,
    src_schema: str | None,
    metadata: MetaData,
    config: MutableMapping[str, Any],
    spec_path: Path | None,
) -> Mapping[str, Any]:
    """
    Update configuration with the specification from a CSV file.

    The specification is a headerless CSV file with columns: Table name,
    Column name (or space-separated list of column names), Generator
    name required, Second choice generator name, Third choice generator
    name, etcetera.
    :param src_dsn: Address of the source database
    :param src_schema: Name of the source database schema to read from
    :param metadata: SQLAlchemy representation of the source database
    :param config: Existing configuration (will be destructively updated)
    :param spec_path: The path of the CSV file containing the specification
    :return: Updated configuration.
    """
    with GeneratorCmd(src_dsn, src_schema, metadata, config) as gc:
        if spec_path is None:
            gc.cmdloop()
            return gc.config
        spec = spec_path.open()
        line_no = 0
        for line in csv.reader(spec):
            line_no += 1
            if line:
                if len(line) != 3:
                    logger.error(
                        "line {0} of file {1} does not have three values",
                        line_no,
                        spec_path,
                    )
                if gc.go_to(f"{line[0]}.{line[1]}"):
                    gc.do_set(line[2])
        gc.do_quit("yes")
        return gc.config
