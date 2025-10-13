"""Table configuration command shell."""
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import sqlalchemy
from sqlalchemy import MetaData

from datafaker.interactive.base import (
    TYPE_LETTER,
    TYPE_PROMPT,
    DbCmd,
    TableEntry,
    TableType,
)


@dataclass
class TableCmdTableEntry(TableEntry):
    """Table entry for the table command shell."""

    old_type: TableType
    new_type: TableType


class TableCmd(DbCmd):
    """Command shell allowing the user to set the type of each table."""

    intro = (
        "Interactive table configuration (ignore,"
        " vocabulary, private, generate or empty). Type ? for help.\n"
    )
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

    def make_table_entry(
        self, table_name: str, table_config: Mapping
    ) -> TableCmdTableEntry | None:
        """
        Make a table entry for the named table.

        :param name: The name of the table.
        :param table: The part of ``config.yaml`` corresponding to this table.
        :return: The newly-constructed table entry.
        """
        if table_config.get("ignore", False):
            return TableCmdTableEntry(table_name, TableType.IGNORE, TableType.IGNORE)
        if table_config.get("vocabulary_table", False):
            return TableCmdTableEntry(
                table_name, TableType.VOCABULARY, TableType.VOCABULARY
            )
        if table_config.get("primary_private", False):
            return TableCmdTableEntry(table_name, TableType.PRIVATE, TableType.PRIVATE)
        if table_config.get("num_rows_per_pass", 1) == 0:
            return TableCmdTableEntry(table_name, TableType.EMPTY, TableType.EMPTY)
        return TableCmdTableEntry(table_name, TableType.GENERATE, TableType.GENERATE)

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

    def _find_entry_by_table_name(self, table_name: str) -> TableCmdTableEntry | None:
        """Get the table entry of the table with the given name."""
        entry = super()._find_entry_by_table_name(table_name)
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
                    to_entry = self._find_entry_by_table_name(ref)
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
                    to_entry = self._find_entry_by_table_name(ref)
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
            self._set_table_index(index)
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
        if not self._set_table_index(self.table_index - 1):
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
        """Set the current table as to be generated, and go to the next table."""
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
        'data 20 ColumnName 30' = report a random twenty entries from
        ColumnName of length at least 30,
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
            where = f"WHERE LENGTH({column}) >= {min_length}"
        with self.sync_engine.connect() as connection:
            result = connection.execute(
                sqlalchemy.text(
                    f"SELECT {column} FROM {self.table_name()}"
                    f" {where} ORDER BY RANDOM() LIMIT {count}"
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
                sqlalchemy.text(
                    f"SELECT * FROM {self.table_name()} ORDER BY RANDOM() LIMIT {count}"
                )
            )
            if result is None:
                self.print("No rows in this table!")
                return
            self.print_results(result)
