from abc import ABC, abstractmethod
import cmd
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import logging
from prettytable import PrettyTable
import re
from sqlalchemy import Column, MetaData, Table, text

from sqlsynthgen.generators import everything_factory, Generator
from sqlsynthgen.utils import create_db_engine

logger = logging.getLogger(__name__)

class TableType(Enum):
    NORMAL = "normal"
    IGNORE = "ignore"
    VOCABULARY = "vocabulary"
    PRIVATE = "private"

TYPE_LETTER = {
    TableType.NORMAL: " ",
    TableType.IGNORE: "I",
    TableType.VOCABULARY: "V",
    TableType.PRIVATE: "P",
}

TYPE_PROMPT = {
    TableType.NORMAL: "(table: {}) ",
    TableType.IGNORE: "(table: {} (ignore)) ",
    TableType.VOCABULARY: "(table: {} (vocab)) ",
    TableType.PRIVATE: "(table: {} (private)) ",
}

@dataclass
class TableEntry:
    name: str


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


class DbCmd(ABC, cmd.Cmd):
    ERROR_NO_MORE_TABLES = "Error: There are no more tables"
    ERROR_ALREADY_AT_START = "Error: Already at the start"
    ERROR_NO_SUCH_TABLE = "Error: '{0}' is not the name of a table in this database"

    @abstractmethod
    def make_table_entry(self, name: str) -> TableEntry:
        ...

    def __init__(self, src_dsn: str, src_schema: str, metadata: MetaData, config: Mapping):
        super().__init__()
        self.config = config
        self.metadata = metadata
        self.table_entries: list[TableEntry] = []
        for name in metadata.tables.keys():
            entry = self.make_table_entry(name)
            if entry is not None:
                self.table_entries.append(entry)
        self.table_index = 0
        self.engine = create_db_engine(src_dsn, schema_name=src_schema)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.engine.dispose()

    def print(self, text: str, *args, **kwargs):
        print(text.format(*args, **kwargs))
    def print_table(self, headings: list[str], rows: list[list[str]]):
        output = PrettyTable()
        output.field_names = headings
        for row in rows:
            output.add_row(row)
        print(output)
    def print_table_by_columns(self, columns: dict[str, list[str]]):
        output = PrettyTable()
        for field_name, data in columns.items():
            output.add_column(field_name, data)
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

    def set_table_index(self, index) -> bool:
        if 0 <= index and index < len(self.table_entries):
            self.table_index = index
            self.set_prompt()
            return True
        return False
    def next_table(self, report="No more tables"):
        if not self.set_table_index(self.table_index + 1):
            self.print(report)
            return False
        return True
    def table_name(self):
        return self.table_entries[self.table_index].name
    def table_metadata(self) -> Table:
        return self.metadata.tables[self.table_name()]


@dataclass
class TableCmdTableEntry(TableEntry):
    old_type: TableType
    new_type: TableType

class TableCmd(DbCmd):
    intro = "Interactive table configuration (ignore, vocabulary or private). Type ? for help.\n"
    prompt = "(tableconf) "
    file = None

    def make_table_entry(self, name: str) -> TableEntry:
        tables = self.config.get("tables", {})
        table = tables.get(name, {})
        if table.get("ignore", False):
            return TableCmdTableEntry(name, TableType.IGNORE, TableType.IGNORE)
        if table.get("vocabulary_table", False):
            return TableCmdTableEntry(name, TableType.VOCABULARY, TableType.VOCABULARY)
        if table.get("primary_private", False):
            return TableCmdTableEntry(name, TableType.PRIVATE, TableType.PRIVATE)
        return TableCmdTableEntry(name, TableType.NORMAL, TableType.NORMAL)

    def __init__(self, src_dsn: str, src_schema: str, metadata: MetaData, config: Mapping):
        super().__init__(src_dsn, src_schema, metadata, config)
        self.config = config
        self.set_prompt()

    def set_prompt(self):
        if self.table_index < len(self.table_entries):
            entry = self.table_entries[self.table_index]
            self.prompt = TYPE_PROMPT[entry.new_type].format(entry.name)
        else:
            self.prompt = "(table) "
    def set_type(self, t_type: TableType):
        if self.table_index < len(self.table_entries):
            entry = self.table_entries[self.table_index]
            entry.new_type = t_type
    def copy_entries(self) -> None:
        tables = self.config.get("tables", {})
        for entry in self.table_entries:
            if entry.old_type != entry.new_type:
                table: dict = tables.get(entry.name, {})
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
                else:
                    table.pop("ignore", None)
                    table.pop("vocabulary_table", None)
                    table.pop("primary_private", None)
                tables[entry.name] = table
        self.config["tables"] = tables

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
    def do_list(self, arg):
        "list the tables with their types"
        for entry in self.table_entries:
            old = entry.old_type
            new = entry.new_type
            becomes = "   " if old == new else "->" + TYPE_LETTER[new]
            self.print("{0}{1} {2}", TYPE_LETTER[old], becomes, entry.name)
    def do_next(self, _arg):
        "'next' = go to the next table, 'next tablename' = go to table 'tablename'"
        if _arg:
            # Find the index of the table called _arg, if any
            index = next((i for i,entry in enumerate(self.table_entries) if entry.name == _arg), None)
            if index is None:
                self.print(self.ERROR_NO_SUCH_TABLE, _arg)
                return
            self.set_table_index(index)
            return
        self.next_table(self.ERROR_NO_MORE_TABLES)
    def complete_next(self, text, line, begidx, endidx):
        return [
            entry.name
            for entry in self.table_entries
            if entry.name.startswith(text)
        ]
    def do_previous(self, _arg):
        "Go to the previous table"
        if not self.set_table_index(self.table_index - 1):
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
    def do_private(self, _arg):
        "Set the current table as a primary private table (such as the table of patients)"
        self.set_type(TableType.PRIVATE)
        self.print("Table {} set to be a primary private table", self.table_name())
        self.next_table()
    def do_normal(self, _arg):
        "Set the current table as neither a vocabulary table nor ignored nor primary private, and go to the next table"
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
    def complete_data(self, text, line, begidx, endidx):
        previous_parts = line[:begidx - 1].split()
        if len(previous_parts) != 2:
            return []
        table_metadata = self.table_metadata()
        return [
            k for k in table_metadata.columns.keys()
            if k.startswith(text)
        ]

    def print_column_data(self, column: str, count: int, min_length: int):
        where = ""
        if 0 < min_length:
            where = "WHERE LENGTH({column}) >= {len}".format(
                column=column,
                len=min_length,
            )
        with self.engine.connect() as connection:
            result = connection.execute(
                text("SELECT {column} FROM {table} {where} ORDER BY RANDOM() LIMIT {count}".format(
                    table=self.table_name(),
                    column=column,
                    count=count,
                    where=where,
                ))
            )
            self.columnize([str(x[0]) for x in result.all()])

    def print_row_data(self, count: int):
        with self.engine.connect() as connection:
            result = connection.execute(
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


@dataclass
class GeneratorInfo:
    column: str
    is_primary_key: bool
    old_name: str | None
    new_name: str | None

@dataclass
class GeneratorCmdTableEntry(TableEntry):
    generators: list[GeneratorInfo]

class GeneratorCmd(DbCmd):
    intro = "Interactive generator configuration. Type ? for help.\n"
    prompt = "(generatorconf) "
    file = None

    def make_table_entry(self, name: str) -> TableEntry:
        tables = self.config.get("tables", {})
        table = tables.get(name, {})
        metadata_table = self.metadata.tables[name]
        columns = set(metadata_table.columns.keys())
        generator_infos: list[GeneratorInfo] = []
        multiple_columns_assigned: dict[str, list[str]] = {}
        for rg in table.get("row_generators", []):
            gen_name = rg.get("name", None)
            if gen_name:
                ca = rg.get("columns_assigned", [])
                single_ca = None
                if isinstance(ca, str):
                    if ca in columns:
                        columns.remove(ca)
                        single_ca = ca
                    else:
                        logger.warning(
                            "table '%s' has '%s' assigned to column '%s' which is not in this table",
                            name, gen_name, ca,
                        )
                else:
                    columns.difference_update(ca)
                    if len(ca) == 1:
                        single_ca = ca
            if single_ca is not None:
                generator_infos.append(GeneratorInfo(
                    column=single_ca,
                    is_primary_key=metadata_table.columns[single_ca].primary_key,
                    old_name=gen_name,
                    new_name=gen_name,
                ))
            else:
                multiple_columns_assigned[gen_name] = ca
        for col in columns:
            generator_infos.append(GeneratorInfo(
                column=col,
                is_primary_key=metadata_table.columns[col].primary_key,
                old_name=None,
                new_name=None,
            ))
        if multiple_columns_assigned:
            self.print(
                "The following mulit-column generators for table {0} are defined in the configuration file and cannot be configured with this command",
                name,
            )
            for (gen_name, cols) in multiple_columns_assigned.items():
                self.print("   {0}: {1}", gen_name, cols)
        if len(generator_infos) == 0:
            return None
        return GeneratorCmdTableEntry(
            name=name,
            generators=generator_infos
        )

    def __init__(self, src_dsn: str, src_schema: str, metadata: MetaData, config: Mapping):
        super().__init__(src_dsn, src_schema, metadata, config)
        self.generator_index = 0
        self.generators_valid_indices = None
        self.set_prompt()

    def set_table_index(self, index):
        ret = super().set_table_index(index)
        if ret:
            self.generator_index = 0
            self.set_prompt()
        return ret

    def previous_table(self):
        ret = self.set_table_index(self.table_index - 1)
        if ret:
            table = self.get_table()
            if table is None:
                self.print("Internal error! table {0} does not have any generators!", self.table_index)
                return False
            self.generator_index = len(table.generators) - 1
        return ret

    def get_table(self) -> GeneratorCmdTableEntry | None:
        if self.table_index < len(self.table_entries):
            return self.table_entries[self.table_index]
        return None

    def get_table_and_generator(self) -> tuple[str | None, GeneratorInfo | None]:
        if self.table_index < len(self.table_entries):
            entry = self.table_entries[self.table_index]
            if self.generator_index < len(entry.generators):
                return (entry.name, entry.generators[self.generator_index])
            return (entry.name, None)
        return (None, None)

    def get_column_name(self) -> str | None:
        (_, generator_info) = self.get_table_and_generator()
        return generator_info.column if generator_info else None

    def column_metadata(self) -> Column | None:
        table = self.table_metadata()
        column_name = self.get_column_name()
        if table is None or column_name is None:
            return None
        return table.columns[column_name]

    def set_prompt(self):
        (table_name, gen_info) = self.get_table_and_generator()
        if table_name is None:
            self.prompt = "(generators) "
            return
        if gen_info is None:
            self.prompt = "({table}) ".format(table=table_name)
            return
        if gen_info.is_primary_key:
            column = f"{gen_info.column}[pk]"
        else:
            column = gen_info.column
        if gen_info.new_name:
            self.prompt = "({table}.{column} ({generator})) ".format(
                table=table_name,
                column=column,
                generator=gen_info.new_name,
            )
        else:
            self.prompt = "({table}.{column}) ".format(
                table=table_name,
                column=column,
            )

    def set_generator(self, generator: str):
        if self.table_index < len(self.table_entries):
            entry = self.table_entries[self.table_index]
            if self.generator_index < len(entry.generators):
                entry.generators[self.generator_index] = generator

    def copy_entries(self) -> None:
        tables = self.config.get("tables", {})
        for entry in self.table_entries:
            # We probably need to reconstruct row_generators. Hmmm.
            # We will need to keep row_generators intact not break them apart like now
            for generator in entry.generators:
                pass
        self.config["tables"] = tables

    def do_quit(self, _arg):
        "Check the updates, save them if desired and quit the configurer."
        count = 0
        for entry in self.table_entries:
            header_shown = False
            for gen in entry.generators:
                if gen.old_name != gen.new_name:
                    if not header_shown:
                        header_shown = True
                        self.print("Table {0}:", entry.name)
                    count += 1
                    self.print(
                        "...changing {0} from {1} to {2}",
                        gen.name,
                        gen.old_name,
                        gen.new_name,
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

    def do_tables(self, arg):
        "list the tables"
        for entry in self.table_entries:
            gen_count = len(entry.generators)
            how_many = "one generator" if gen_count == 1 else f"{gen_count} generators"
            self.print("{0} ({1})", entry.name, how_many)

    def do_list(self, arg):
        "list the generators in the current table"
        if len(self.table_entries) <= self.table_index:
            self.print("Error: no table {0}", self.table_index)
            return
        for gen in self.table_entries[self.table_index].generators:
            old = "" if gen.old_name is None else gen.old_name
            if gen.old_name == gen.new_name:
                becomes = ""
                if old == "":
                    old = "(not set)"
            elif gen.new_name is None:
                becomes = "(delete)"
            else:
                becomes = f"->{gen.new_name}"
            primary = "[primary-key]" if gen.is_primary_key else ""
            self.print("{0}{1}{2} {3}", old, becomes, primary, gen.column)

    def do_columns(self, _arg):
        "Report the column names"
        self.columnize(self.table_metadata().columns.keys())

    def do_next(self, _arg):
        "Go to the next generator"
        table = self.get_table()
        if table is None:
            self.print("No more tables")
        next_gi = self.generator_index + 1
        if next_gi == len(table.generators):
            self.next_table()
            return
        self.generator_index = next_gi
        self.set_prompt()

    def do_previous(self, _arg):
        "Go to the previous generator"
        if self.generator_index == 0:
            self.previous_table()
        else:
            self.generator_index -= 1
        self.set_prompt()

    def get_generator_proposals(self) -> list[Generator]:
        if (self.table_index, self.generator_index) != self.generators_valid_indices:
            self.generators = None
        if self.generators is None:
            column = self.column_metadata()
            if column is None:
                logger.error("No such column")
                return []
            gens = everything_factory.get_generators(column, self.engine)
            gens.sort(key=lambda g: g.fit(9999))
            self.generators = gens
        return self.generators

    def do_compare(self, arg: str):
        """
        Compare the real data with some generators.

        'compare': just look at some source data from this column.
        'compare 5 6 10': compare a sample of the source data with a sample
        from generators 5, 6 and 10. You can find out which numbers
        correspond to which generators using the 'propose' command.
        """
        args = arg.split()
        limit = 20
        comparison = {
            "source": self.get_column_data(limit, to_str=str),
        }
        gens: list[Generator] = self.get_generator_proposals()
        table_name = self.table_name()
        for argument in args:
            if argument.isnumeric():
                n = int(argument)
                if 0 < n and n <= len(gens):
                    gen = gens[n - 1]
                    comparison[f"{n}. {gen.function_name()}"] = gen.generate_data(limit)
                    self.print_values_queried(table_name, n, gen)
        self.print_table_by_columns(comparison)

    def print_values_queried(self, table_name: str, n: int, gen: Generator):
        """
        Print the values queried from the database for this generator.
        """
        if not gen.select_aggregate_clauses() and not gen.custom_queries():
            self.print(
                "{0}. {1} requires no data from the source database.",
                n,
                gen.function_name(),
            )
        else:
            self.print(
                "{0}. {1} requires the following data from the source database:",
                n,
                gen.function_name(),
            )
            self.print_select_aggregate_query(table_name, gen)
            self.print_custom_queries(gen)

    def print_custom_queries(self, gen: Generator) -> None:
        """
        Print all the custom queries and all the values they get in this case.
        """
        cqs = gen.custom_queries()
        if not cqs:
            return
        kwa = gen.actual_kwargs()
        cq_key2args = {}
        src_stat_re = re.compile(f'SRC_STATS\\["([^"]+)"\\]\\["([^"]+)"\\]')
        for argname, src_stat in gen.nominal_kwargs().items():
            if argname in kwa:
                src_stat_groups = src_stat_re.match(src_stat)
                if src_stat_groups:
                    cq_key = src_stat_groups.group(1)
                    if cq_key not in cq_key2args:
                        cq_key2args[cq_key] = []
                    cq_key2args[cq_key].append(kwa[argname])
        for cq_key, cq in cqs.items():
            self.print("{0}; providing the following values: {1}", cq, cq_key2args[cq_key])

    def print_select_aggregate_query(self, table_name, gen) -> None:
        """
        Prints the select aggregate query and all the values it gets in this case.
        """
        sacs = gen.select_aggregate_clauses()
        if not sacs:
            return
        kwa = gen.actual_kwargs()
        clauses = [
                                f"{q} AS {n}"
                                for n, q in sacs.items()
                            ]
        vals = []
        src_stat2kwarg = { v: k for k, v in gen.nominal_kwargs().items() }
        for n in sacs.keys():
            src_stat = f'SRC_STATS["auto__{table_name}"]["{n}"]'
            if src_stat in src_stat2kwarg:
                ak = src_stat2kwarg[src_stat]
                if ak in kwa:
                    vals.append(kwa[ak])
                else:
                    vals.append("(actual_kwargs() does not report)")
            else:
                vals += "(unused)"
        select_q = f"SELECT {', '.join(clauses)} FROM {table_name}"
        self.print("{0}; providing the following values: {1}", select_q, vals)

    def get_column_data(self, count: int, to_str=repr, min_length: int = 0):
        column = self.get_column_name()
        where = ""
        if 0 < min_length:
            where = "WHERE LENGTH({column}) >= {len}".format(
                column=column,
                len=min_length,
            )
        with self.engine.connect() as connection:
            result = connection.execute(
                text("SELECT {column} FROM {table} {where} ORDER BY RANDOM() LIMIT {count}".format(
                    table=self.table_name(),
                    column=column,
                    count=count,
                    where=where,
                ))
            )
            return [to_str(x[0]) for x in result.all()]

    def do_propose(self, arg):
        """
        Display a list of possible generators for this column.

        They will be listed in order of fit, the most likely matches first.
        The results can be compared (against a sample of the real data in
        the column and against each other) with the 'compare' command.
        """
        limit = 5
        gens = self.get_generator_proposals()
        sample = self.get_column_data(limit)
        self.print("Sample of actual source data: {0}...", ",".join(sample))
        for index, gen in enumerate(gens):
            fit = gen.fit()
            fit_s = "(no fit)" if fit is None else f"(fit: {fit:.0f})"
            self.print(
                "{index}. {name}: {fit} {sample} ...",
                index=index + 1,
                name=gen.function_name(),
                fit=fit_s,
                sample=", ".join(map(repr, gen.generate_data(limit)))
            )


def update_config_generators(src_dsn: str, src_schema: str, metadata: MetaData, config: Mapping):
    with GeneratorCmd(src_dsn, src_schema, metadata, config) as gc:
        gc.cmdloop()
        return gc.config
