import cmd
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Self

from prettytable import PrettyTable
from sqlalchemy import MetaData, Table, Column, text

from sqlsynthgen.utils import create_db_engine

logger = logging.getLogger(__name__)

class TableType(Enum):
    NORMAL = "normal"
    IGNORE = "ignore"
    VOCABULARY = "vocabulary"

@dataclass
class TableEntry:
    name: str
    old_type: TableType
    new_type: TableType
    @classmethod
    def make(_cls, name: str, config: Mapping) -> Self:
        tables = config.get("tables", {})
        table = tables.get(name, {})
        if table.get("ignore", False):
            return TableEntry(name, TableType.IGNORE, TableType.IGNORE)
        if table.get("vocabulary_table", False):
            return TableEntry(name, TableType.VOCABULARY, TableType.VOCABULARY)
        return TableEntry(name, TableType.NORMAL, TableType.NORMAL)


class AskSaveCmd(cmd.Cmd):
    intro = "Do you want to save this configuration?"
    prompt = "(yes/no/cancel) "
    file = None
    def __init__(self):
        super().__init__()
        self.result = ""
    def do_yes(self, _arg):
        self.result = "yes"
        return True
    def do_no(self, _arg):
        self.result = "no"
        return True
    def do_cancel(self, _arg):
        self.result = "cancel"
        return True


class TableCmd(cmd.Cmd):
    intro = "Interactive table configuration (ignore or vocabulary). Type ? for help.\n"
    prompt = "(tableconf) "
    file = None
    ERROR_NO_MORE_TABLES = "Error: There are no more tables"
    ERROR_ALREADY_AT_START = "Error: Already at the start"
    ERROR_NO_SUCH_TABLE = "Error: '{0}' is not the name of a table in this database"

    def __init__(self, src_dsn: str, src_schema: str, metadata: MetaData, config: Mapping):
        super().__init__()
        self.table_entries: list[TableEntry] = [
            TableEntry.make(name, config)
            for name in metadata.tables.keys()
        ]
        self.table_index = 0
        self.config = config
        self.metadata = metadata
        self.set_prompt()
        self.engine = create_db_engine(src_dsn, schema_name=src_schema)
        self.connection = self.engine.connect()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.close()
        self.engine.dispose()

    def set_prompt(self):
        if self.table_index < len(self.table_entries):
            entry = self.table_entries[self.table_index]
            if entry.new_type == TableType.IGNORE:
                self.prompt = "(table: {} (ignored)) ".format(entry.name)
            elif entry.new_type == TableType.VOCABULARY:
                self.prompt = "(table: {} (vocab)) ".format(entry.name)
            else:
                self.prompt = "(table: {}) ".format(entry.name)
        else:
            self.prompt = "(table)"
    def set_type(self, t_type: TableType):
        if self.table_index < len(self.table_entries):
            entry = self.table_entries[self.table_index]
            entry.new_type = t_type
    def set_index(self, index) -> bool:
        if 0 <= index and index < len(self.table_entries):
            self.table_index = index
            self.set_prompt()
            return True
        return False
    def next_table(self, report="No more tables"):
        if not self.set_index(self.table_index + 1):
            self.print(report)
    def table_name(self):
        return self.table_entries[self.table_index].name
    def table_metadata(self) -> Table:
        return self.metadata.tables[self.table_name()]
    def copy_entries(self) -> None:
        tables = self.config.get("tables", {})
        for entry in self.table_entries:
            if entry.old_type != entry.new_type:
                table: dict = tables.get(entry.name, {})
                if entry.new_type == TableType.IGNORE:
                    table["ignore"] = True
                    table.pop("vocabulary_table", None)
                elif entry.new_type == TableType.VOCABULARY:
                    table.pop("ignore", None)
                    table["vocabulary_table"] = True
                else:
                    table.pop("ignore", None)
                    table.pop("vocabulary_table", None)
                tables[entry.name] = table
        self.config["tables"] = tables

    def print(self, text: str, *args, **kwargs):
        print(text.format(*args, **kwargs))
    def print_table(self, headings: list[str], rows: list[list[str]]):
        output = PrettyTable()
        output.field_names = headings
        for row in rows:
            output.add_row(row)
        print(output)
    def print_results(self, result):
        self.print_table(
            list(result.keys()),
            [list(row) for row in result.all()]
        )
    def ask_save(self):
        ask = AskSaveCmd()
        ask.cmdloop()
        return ask.result

    def do_quit(self, _arg):
        "Check the updates, save them if desired and quit the configurer."
        count = 0
        for entry in self.table_entries:
            if entry.old_type != entry.new_type:
                count += 1
                self.print(
                    "Changing {0} from {1} to {2}",
                    entry.name,
                    entry.old_type.value,
                    entry.new_type.value,
                )
        if count == 0:
            self.print("There are no changes.")
            return True
        reply = self.ask_save()
        if reply == "yes":
            self.copy_entries()
            return True
        if reply == "no":
            return True
        return False
    def do_next(self, _arg):
        "'next' = go to the next table, 'next tablename' = go to table 'tablename'"
        if _arg:
            # Find the index of the table called _arg, if any
            index = next((i for i,entry in enumerate(self.table_entries) if entry.name == _arg), None)
            if index is None:
                self.print(self.ERROR_NO_SUCH_TABLE, _arg)
                return
            self.set_index(index)
            return
        self.next_table(self.ERROR_NO_MORE_TABLES)
    def do_previous(self, _arg):
        "Go to the previous table"
        if not self.set_index(self.table_index - 1):
            self.print(self.ERROR_ALREADY_AT_START)
    def do_ignore(self, _arg):
        "Set the current table as ignored, and go to the next table"
        self.set_type(TableType.IGNORE)
        self.print("Table {} set as ignored", self.table_name())
        self.next_table()
    def do_vocabulary(self, _arg):
        "Set the current table as a vocabulary table, and go to the next table"
        self.set_type(TableType.VOCABULARY)
        self.print("Table {} set to be a vocabulary table", self.table_name())
        self.next_table()
    def do_reset(self, _arg):
        "Set the current table as neither a vocabulary table nor ignored, and go to the next table"
        self.set_type(TableType.NORMAL)
        self.print("Table {} reset", self.table_name())
        self.next_table()
    def do_columns(self, _arg):
        "Report the column names"
        self.columnize(self.table_metadata().columns.keys())
    def do_data(self, arg: str):
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
        if arg_index < len(args) and args[arg_index].isnumeric():
            number = int(args[arg_index])
            arg_index += 1
        if arg_index < len(args) and args[arg_index] in table_metadata.columns:
            column = args[arg_index]
            arg_index += 1
        if arg_index < len(args) and args[arg_index].isnumeric():
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

    def print_column_data(self, column: str, count: int, min_length: int):
        where = ""
        if 0 < min_length:
            where = "WHERE LENGTH({column}) >= {len}".format(
                column=column,
                len=min_length,
            )
        result = self.connection.execute(
            text("SELECT {column} FROM {table} {where} ORDER BY RANDOM() LIMIT {count}".format(
                table=self.table_name(),
                column=column,
                count=count,
                where=where,
            ))
        )
        self.columnize([x[0] for x in result.all()])

    def print_row_data(self, count: int):
        result = self.connection.execute(
            text("SELECT * FROM {table} ORDER BY RANDOM() LIMIT {count}".format(
                table=self.table_name(),
                count=count,
            ))
        )
        if result is None:
            self.print("No rows in this table!")
            return
        self.print_results(result)

def update_config_tables(src_dsn: str, src_schema: str, metadata: MetaData, config: Mapping):
    with TableCmd(src_dsn, src_schema, metadata, config) as tc:
        tc.cmdloop()
        return tc.config
