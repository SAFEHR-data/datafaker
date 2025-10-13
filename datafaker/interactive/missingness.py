"""Missingness configuration shell."""
from dataclasses import dataclass
from collections.abc import Iterable, Mapping, MutableMapping
import re
from typing import cast

from sqlalchemy import MetaData

from datafaker.interactive.base import DbCmd, TableEntry

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
        result_names = ", ".join([f"{c}__is_null" for c in column_names])
        column_is_nulls = ", ".join(
            [f"{c} IS NULL AS {c}__is_null" for c in column_names]
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
                    if not isinstance(query, str):
                        return None
                    return (query, src_stat.get("comment", ""))
        return None

    def make_table_entry(
        self, table_name: str, table_config: Mapping
    ) -> MissingnessCmdTableEntry | None:
        """
        Make a table entry for a particular table.

        :param name: The name of the table to make an entry for.
        :param table: The part of ``config.yaml`` relating to this table.
        :return: The newly-constructed table entry.
        """
        if table_config.get("ignore", False):
            return None
        if table_config.get("vocabulary_table", False):
            return None
        if table_config.get("num_rows_per_pass", 1) == 0:
            return None
        mgs = table_config.get("missingness_generators", [])
        old = None
        nonnull_columns = self.get_nonnull_columns(table_name)
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
            if isinstance(mg_name, str):
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
            name=table_name,
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

    def _find_entry_by_table_name(
        self, table_name: str
    ) -> MissingnessCmdTableEntry | None:
        """Find the table entry given the table name."""
        entry = super()._find_entry_by_table_name(table_name)
        if entry is None:
            return None
        return cast(MissingnessCmdTableEntry, entry)

    def set_prompt(self) -> None:
        """Set the prompt according to the current table and missingness."""
        if self.table_index < len(self.table_entries):
            entry: MissingnessCmdTableEntry = self.table_entries[self.table_index]
            nt = entry.new_type
            if nt is None:
                self.prompt = f"(missingness for {entry.name}) "
            else:
                self.prompt = f"(missingness for {entry.name}: {nt.name}) "
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
                src_stat_key = f"missing_auto__{entry.name}__0"
                table["missingness_generators"] = [
                    {
                        "name": entry.new_type.name,
                        "kwargs": {
                            "patterns": f'SRC_STATS["{src_stat_key}"]["results"]'
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
            desc = new if old == new else f"{old}->{new}"
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
            self._set_table_index(index)
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
        if not self._set_table_index(self.table_index - 1):
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
                (
                    "Error: sampled can be used alone or with"
                    " an integer argument. {0} is not permitted"
                ),
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
            (
                "The missingness patterns and how often they appear in a"
                f" sample of {count} from table {entry.name}"
            ),
        )
        self.print("Table {} set to sampled missingness", self.table_name())
        self.next_table()

    def do_none(self, _arg: str) -> None:
        """Set the current table to have no missingness, and go to the next table."""
        self._set_none()
        self.print("Table {} set to have no missingness", self.table_name())
        self.next_table()
