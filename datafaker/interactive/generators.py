"""Generator configuration shell."""  # pylint: disable=too-many-lines
import functools
import re
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

import sqlalchemy
from sqlalchemy import Column, MetaData

from datafaker.generators import everything_factory
from datafaker.generators.base import Generator, PredefinedGenerator
from datafaker.interactive.base import DbCmd, TableEntry, fk_column_name, or_default
from datafaker.utils import (
    get_columns_assigned,
    get_row_generators,
    logger,
    primary_private_fks,
    split_column_full_name,
    table_is_private,
)


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


# pylint: disable=too-many-public-methods
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
        self, table_name: str, table_config: Mapping
    ) -> GeneratorCmdTableEntry | None:
        """
        Make a table entry.

        :param table_name: The name of the table.
        :param table: The portion of the ``config.yaml`` file describing this table.
        :return: The newly constructed table entry, or None if this table is to be ignored.
        """
        if table_config.get("ignore", False):
            return None
        if table_config.get("vocabulary_table", False):
            return None
        if table_config.get("num_rows_per_pass", 1) == 0:
            return None
        columns = [
            str(colname) for colname in self.metadata.tables[table_name].columns.keys()
        ]
        column_set = frozenset(columns)
        columns_assigned_so_far: set[str] = set()

        new_generator_infos: list[GeneratorInfo] = []
        for gen_name, rg in get_row_generators(table_config):
            colset: set[str] = set(get_columns_assigned(rg))
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
            actual_collist = [c for c in columns if c in colset]
            if actual_collist:
                new_generator_infos.append(
                    GeneratorInfo(
                        columns=actual_collist.copy(),
                        gen=PredefinedGenerator(table_name, rg, self.config),
                    )
                )
                columns_assigned_so_far |= colset
        old_generator_infos = [
            GeneratorInfo(columns=gi.columns.copy(), gen=gi.gen)
            for gi in new_generator_infos
        ]
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
        self.generators: list[Generator] | None = None
        self.generator_index = 0
        self.generators_valid_columns: Optional[tuple[int, list[str]]] = None
        self.set_prompt()

    @property
    def table_entries(self) -> list[GeneratorCmdTableEntry]:
        """Get the talbe entries, cast to ``GeneratorCmdTableEntry``."""
        return cast(list[GeneratorCmdTableEntry], self._table_entries)

    def _find_entry_by_table_name(
        self, table_name: str
    ) -> GeneratorCmdTableEntry | None:
        """
        Find the table entry by name.

        :param table_name: The name of the table to find.
        :return: The table entry, or None if no such table name exists.
        """
        entry = super()._find_entry_by_table_name(table_name)
        if entry is None:
            return None
        return cast(GeneratorCmdTableEntry, entry)

    def _set_table_index(self, index: int) -> bool:
        """
        Move to a new table.

        :param index: table index to move to.
        """
        ret = super()._set_table_index(index)
        if ret:
            self.generator_index = 0
            self.set_prompt()
        return ret

    def _previous_table(self) -> bool:
        """
        Move to the table before the current one.

        :return: True if there is a previous table to go to.
        """
        ret = self._set_table_index(self.table_index - 1)
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

    def _get_table_and_generator(self) -> tuple[str | None, GeneratorInfo | None]:
        """Get a pair; the table name then the generator information."""
        if self.table_index < len(self.table_entries):
            entry = self.table_entries[self.table_index]
            if self.generator_index < len(entry.new_generators):
                return (entry.name, entry.new_generators[self.generator_index])
            return (entry.name, None)
        return (None, None)

    def _get_column_names(self) -> list[str]:
        """Get the (unqualified) names for all the current columns."""
        (_, generator_info) = self._get_table_and_generator()
        return generator_info.columns if generator_info else []

    def _column_metadata(self) -> list[Column]:
        """Get the metadata for all the current columns."""
        table = self.table_metadata()
        if table is None:
            return []
        return [table.columns[name] for name in self._get_column_names()]

    def set_prompt(self) -> None:
        """Set the prompt according to the current table, column and generator."""
        (table_name, gen_info) = self._get_table_and_generator()
        if table_name is None:
            self.prompt = "(generators) "
            return
        if gen_info is None:
            self.prompt = f"({table_name}) "
            return
        table = self.table_metadata()
        columns = [
            c + "[pk]" if table.columns[c].primary_key else c for c in gen_info.columns
        ]
        gen = f" ({gen_info.gen.name()})" if gen_info.gen else ""
        self.prompt = f"({table_name}.{','.join(columns)}{gen}) "

    def _remove_auto_src_stats(self) -> list[MutableMapping[str, Any]]:
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
                                "comments": cq.get("comments", []),
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

    def do_tables(self, _arg: str) -> None:
        """List the tables."""
        for t_entry in self.table_entries:
            entry = cast(GeneratorCmdTableEntry, t_entry)
            gen_count = len(entry.new_generators)
            how_many = "one generator" if gen_count == 1 else f"{gen_count} generators"
            self.print("{0} ({1})", entry.name, how_many)

    def do_list(self, _arg: str) -> None:
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
        for cm in self._column_metadata():
            self.print(
                "Column {0} in table {1} has type {2} ({3}).",
                cm.name,
                cm.table.name,
                str(cm.type),
                "nullable" if cm.nullable else "not nullable",
            )
            if cm.primary_key:
                self.print(
                    "It is a primary key, which usually does not"
                    " need a generator (it will auto-increment)"
                )
            if cm.foreign_keys:
                fk_names = [fk_column_name(fk) for fk in cm.foreign_keys]
                self.print(
                    "It is a foreign key referencing column {0}", ", ".join(fk_names)
                )
                if len(fk_names) == 1 and not cm.primary_key:
                    self.print(
                        "You do not need a generator if you just want"
                        " a uniform choice over the referenced table's rows"
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
        (first_part, last_part) = split_column_full_name(target)
        gen_index: int | None = None
        if first_part:
            # table.column
            table_index = self._get_table_index(first_part)
            if table_index is None:
                self.print(self.ERROR_NO_SUCH_TABLE, first_part)
                return False
            gen_index = self._get_generator_index(table_index, last_part)
            if gen_index is None:
                self.print(self.ERROR_NO_SUCH_COLUMN, last_part)
                return False
        else:
            # just table or column
            table_index = self._get_table_index(last_part)
            gen_index = 0
            if table_index is None:
                # not table, perhaps it's column
                gen_index = self._get_generator_index(self.table_index, last_part)
                if gen_index is None:
                    # it's neither
                    self.print(self.ERROR_NO_SUCH_TABLE_OR_COLUMN, last_part)
                    return False
        if table_index is not None:
            self._set_table_index(table_index)
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
        (first_part, last_part) = split_column_full_name(text)
        if first_part:
            table_index = self._get_table_index(first_part)
            if table_index is None:
                return []
            table_entry = self.table_entries[table_index]
            return [
                f"{first_part}.{column}"
                for gen in table_entry.new_generators
                for column in gen.columns
                if column.startswith(last_part)
            ]
        table_names = [
            entry.name
            for entry in self.table_entries
            if entry.name.startswith(last_part)
        ]
        if last_part in table_names:
            table_names.append(f"{last_part}.")
        current_table = self.get_table()
        if current_table:
            column_names = [
                col
                for gen in current_table.new_generators
                for col in gen.columns
                if col.startswith(last_part)
            ]
        else:
            column_names = []
        return table_names + column_names

    def do_previous(self, _arg: str) -> None:
        """Go to the previous generator."""
        if self.generator_index == 0:
            self._previous_table()
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
            self._get_column_names(),
        )

    def _get_generator_proposals(self) -> list[Generator]:
        """Get a list of acceptable generators, sorted by decreasing fit to the actual data."""
        if not self._generators_valid():
            self.generators = None
        if self.generators is None:
            columns = self._column_metadata()
            gens = everything_factory(self.config).get_generators(
                columns, self.sync_engine
            )
            sorted_gens = sorted(gens, key=lambda g: g.fit(9999))
            self.generators = sorted_gens
            self.generators_valid_columns = (
                self.table_index,
                self._get_column_names().copy(),
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
                if 0 < n <= len(gens):
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
        if isinstance(nominal, str):
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
        elif isinstance(nominal, Sequence) and isinstance(actual, Sequence):
            for i in range(min(len(nominal), len(actual))):
                self._get_custom_queries_from(out, nominal[i], actual[i])
        elif isinstance(nominal, Mapping) and isinstance(actual, Mapping):
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
                    (
                        "nominal_kwargs for %s does not have a value"
                        ' SRC_STATS["auto__%s"]["results"][0]["%s"]'
                    ),
                    gen.name(),
                    table_name,
                    n,
                )
        select_q = self._get_aggregate_query([gen], table_name)
        self.print("{0}; providing the following values: {1}", select_q, vals)

    def _get_column_data(
        self, count: int, to_str: Callable[[Any], str] = repr
    ) -> list[list[str]]:
        columns = self._get_column_names()
        columns_string = ", ".join(columns)
        pred = " AND ".join(f"{column} IS NOT NULL" for column in columns)
        with self.sync_engine.connect() as connection:
            result = connection.execute(
                sqlalchemy.text(
                    f"SELECT {columns_string} FROM {self.table_name()}"
                    f" WHERE {pred} ORDER BY RANDOM() LIMIT {count}"
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
        (table, gen_info) = self._get_table_and_generator()
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

    def merge_columns(self, arg: str) -> bool:
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
            return False
        cols_available = functools.reduce(
            lambda x, y: x | y,
            [frozenset(gen.columns) for gen in table_entry.new_generators],
        )
        cols_to_merge = frozenset(cols)
        unknown_cols = cols_to_merge - cols_available
        if unknown_cols:
            for uc in unknown_cols:
                self.print(self.ERROR_NO_SUCH_COLUMN, uc)
            return False
        gen_info = table_entry.new_generators[self.generator_index]
        stated_current_columns = cols_to_merge & frozenset(gen_info.columns)
        if stated_current_columns:
            for c in stated_current_columns:
                self.print(self.ERROR_COLUMN_ALREADY_MERGED, c)
            return False
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
        return True

    def do_merge(self, arg: str) -> None:
        """Add this column(s) to the specified column(s), so one generator covers them all."""
        self.merge_columns(arg)

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

    def get_current_columns(self) -> set[str]:
        """Get the current colums."""
        table_entry: GeneratorCmdTableEntry | None = self.get_table()
        if table_entry is None:
            return set()
        gen_info = table_entry.new_generators[self.generator_index]
        return set(gen_info.columns)

    def set_merged_columns(self, first_col: str, other_cols: str) -> bool:
        """
        Merge columns, after unmerging everything we don't want.

        :param first_col: The first column we want in the merge, must already
        be in this column set.
        :param other_cols: all the columns we want merged other than
        first_col, in order, space-separated.
        :return: True if the merge worked, false if there was an error
        """
        existing = self.get_current_columns()
        existing.discard(first_col)
        for to_remove in existing:
            self.do_unmerge(to_remove)
        return self.merge_columns(other_cols)


def try_setting_generator(gc: GeneratorCmd, gens: Iterable[str]) -> bool:
    """
    Set the current generator by name if possible.

    :param gc: The interactive ``GeneratorCmd`` to use.
    :param gens: A list of names of generators to try, in order.
    :return: True if one of the generators was successfully set, False otherwise.
    """
    for gen in gens:
        new_gen = gc.get_proposed_generator_by_name(gen)
        if new_gen is not None:
            gc.set_generator(new_gen)
            return True
    return False
