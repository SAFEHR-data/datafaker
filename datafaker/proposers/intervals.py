"""Proposers for date intervals."""
import datetime
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import Column, Dialect, Engine, ForeignKey, MetaData, func, select
from sqlalchemy.types import Date, DateTime

from datafaker.db_utils import get_fk_column_between
from datafaker.dialects import SecondsDifference, StdDev
from datafaker.proposers.base import Buckets, Proposer, ProposerFactory, get_column_type
from datafaker.providers import AnchoredProvider
from datafaker.utils import get_property

RelatedColumn = tuple[ForeignKey | None, Column]


def _set_roles_for_column(
    out: dict[str, list[RelatedColumn]],
    fk: ForeignKey | None,
    column: Column,
    column_config: Mapping,
) -> None:
    """
    Set new entries in ``out`` based on the roles ``column`` has.

    :param out: Mapping to be updated of role to related columns. A related
        colomn is a pair ``(fk, fcol)`` where ``fcol`` is the column that has
        that role and ``fk`` is the foreign key in the table ``column`` appears
        in that points to the table ``fcol`` appears in, or None if ``column``
        and ``fcol`` are in the same table.
    :param fk: Foreign key to the table ``column`` appears in (appears in
        new entries set in ``out``).
    :param column: The column to be checked for roles.
    :param column_config: The ``tables: <table-name>: columns:`` section
        of the ``config.yaml`` file.
    """
    roles: list[Any] = get_property(column_config, [column.name, "roles"], [])
    for role in roles:
        pair: RelatedColumn = (fk, column)
        if role not in out:
            out[role] = [pair]
        else:
            out[role].append(pair)


def _get_roles(
    config: Mapping,
    column: Column,
) -> dict[str, list[RelatedColumn]]:
    """
    Work out where the roles are relative to this table.

    :param config: The configuration from ``config.yaml``.
    :param column: The column we are to propose for.
    :return: dictionary of ``role_name`` -> ``(fk or None, column_name)``
        where ``fk`` is the actual foreign key from the table, and ``None``
        means a column from the same table as the input column(s)
        has the required role.
    """
    table = column.table
    tables_config: dict[str, Any] = get_property(config, "tables", {})
    table_conf: dict[str, Any] = get_property(
        tables_config, [str(table.name), "columns"], {}
    )
    role_to_fk_columns: dict[str, list[RelatedColumn]] = {}
    for col in table.columns:
        _set_roles_for_column(role_to_fk_columns, None, col, table_conf)
        # look for roles in related tables
        if col.foreign_keys:
            fk = list(col.foreign_keys)[0]
            target_table = fk.column.table
            ft_conf: dict[str, Any] = get_property(
                tables_config, [str(target_table.name), "columns"], {}
            )
            for fcol in target_table.columns:
                _set_roles_for_column(role_to_fk_columns, fk, fcol, ft_conf)
    return role_to_fk_columns


class DateAfterProposer(Proposer):
    """Proposer that proposes dates that are after a preexisting date."""

    # pylint: disable=too-many-arguments too-many-positional-arguments
    def __init__(
        self,
        metadata: MetaData,
        sd: float,
        mean: float,
        column: Column,
        anchor: Column,
        dialect: Dialect,
        buckets: Buckets | None = None,
    ):
        """
        Initialise a date after proposer.

        :param column_name: The name of the column (in the same table as the
         output) in which to find the anchor.
        """
        super().__init__()
        self._sd = sd
        self._mean = mean
        self._anchor = anchor
        self._column = column
        self._dialect = dialect
        self._provider = AnchoredProvider(metadata=metadata)
        if buckets is None:
            self._fit = None
            return
        dummy_anchor = datetime.datetime.fromisoformat("1970-01-01")
        samples = [(d - dummy_anchor).total_seconds() for d in self.generate_data(400)]
        self._fit = buckets.fit_from_values(samples)

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        if self._column.table == self._anchor.table:
            return "generic.anchored_provider.normal_date"
        return "generic.anchored_provider.normal_date_fk"

    def name(self) -> str:
        """Get the name of the generator."""
        fname = self.function_name()
        aname = self._anchor.name
        atable = self._anchor.table
        if atable == self._column.table:
            return f"{fname} [anchored to {aname}]"
        return f"{fname} [anchored to {aname} of table {atable.name}]"

    def nominal_kwargs(self) -> dict[str, Any]:
        """Get the arguments to be entered into ``config.yaml``."""
        column = self._column
        anchor = self._anchor
        (fk_col, fk) = get_fk_column_between(column.table, anchor.table)
        if fk_col is None or fk is None:
            return {
                "mean_seconds": (
                    f'SRC_STATS["auto__{column.table.name}"]'
                    f'["results"][0]["mean__{column.name}"]'
                ),
                "sd_seconds": (
                    f'SRC_STATS["auto__{column.table.name}"]'
                    f'["results"][0]["stddev__{column.name}"]'
                ),
                "anchor": f'GENERATED_ROW["{anchor.name}"]',
            }
        key = f"auto__interval__{column.table.name}__{column.name}"
        return {
            "dst_db_conn": "dst_db_conn",
            "anchor_column": f'"{anchor.name}"',
            "table": f'"{anchor.table.name}"',
            "mean_seconds": (f'SRC_STATS["{key}"]["results"][0]["mean"]'),
            "sd_seconds": (f'SRC_STATS["{key}"]["results"][0]["sd"]'),
            "anchor_row": f'GENERATED_ROW["{fk_col.name}"]',
            "on_column": f'"{fk.column.name}"',
        }

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        column = self._column
        anchor = self._anchor
        (_, fk) = get_fk_column_between(column.table, anchor.table)
        if fk is None:
            return {
                "mean_seconds": self._sd,
                "sd_seconds": self._mean,
                "anchor": "1970-01-01",
            }
        return {
            "anchor_column": anchor.name,
            "table": anchor.table.name,
            "mean_seconds": self._sd,
            "sd_seconds": self._mean,
            "on_column": fk.column.name,
        }

    def select_aggregate_clauses(self) -> dict[str, dict[str, str]]:
        """
        Get the query fragments the generators need to call.

        This is for anchors in the same table.
        """
        column = self._column
        anchor = self._anchor
        if column.table != anchor.table:
            return {}
        mean_q = func.avg(SecondsDifference(column, anchor))
        sd_q = StdDev(SecondsDifference(column, anchor))

        return {
            f"mean__{column.name}": {
                "clause": str(
                    mean_q.compile(
                        dialect=self._dialect, compile_kwargs={"literal_binds": True}
                    )
                ),
                "comment": (
                    f"Mean of interval between {anchor.name} of {anchor.table.name}"
                    f" and {column.name} from table {column.table.name}."
                ),
            },
            f"stddev__{column.name}": {
                "clause": str(
                    sd_q.compile(
                        dialect=self._dialect, compile_kwargs={"literal_binds": True}
                    )
                ),
                "comment": (
                    f"Standard deviation of interval between {anchor.name}"
                    f" and {column.name} from table {column.table.name}."
                ),
            },
        }

    def custom_queries(self) -> dict[str, dict[str, Any]]:
        """
        Get the query fragments the generators need to call.

        This is for anchors in a related table.
        """
        column = self._column
        anchor = self._anchor
        if column.table == anchor.table:
            return {}
        query = (
            select(
                func.avg(SecondsDifference(column, anchor)).label("mean"),
                StdDev(SecondsDifference(column, anchor)).label("sd"),
            )
            .select_from(column.table)
            .join(anchor.table)
        )
        return {
            f"auto__interval__{column.table.name}__{column.name}": {
                "comments": [
                    "Mean and standard deviation of the length of time between"
                    f" column {anchor.name} of table {anchor.table.name}"
                    f" and column {column.name} of table {column.table.name}."
                ],
                "query": str(
                    query.compile(
                        dialect=self._dialect,
                        compile_kwargs={"literal_binds": True},
                    )
                ),
            }
        }

    def fit(self, default: float = -1) -> float:
        """Get this generator's fit against the real data."""
        return default if self._fit is None else self._fit

    def generate_data(self, count: int) -> list[datetime.datetime]:
        """Generate ``count`` random data points for this column."""
        dummy_anchor = datetime.datetime.fromisoformat("1970-01-01")
        return [
            self._provider.normal_date(self._sd, self._mean, dummy_anchor)
            for _ in range(count)
        ]


class DateAfterProposerFactory(ProposerFactory):
    """Makes proposers for dates after another anchor date."""

    def __init__(self, config: Mapping, metadata: MetaData):
        """Initialize ``DateAfterProposerFactory``."""
        super().__init__()
        self._config = config
        self._metadata = metadata

    def make_date_after_proposers(
        self, engine: Engine, column: Column, anchor: Column
    ) -> list[DateAfterProposer]:
        """Create a ``DateAfterProposer`` object."""
        query = select(
            func.avg(SecondsDifference(column, anchor)).label("mean"),
            StdDev(SecondsDifference(column, anchor)).label("sd"),
        ).select_from(column.table)
        join_tables = []
        if anchor.table != column.table:
            join_tables = [anchor.table]
            query = query.join(anchor.table)
        with engine.connect() as connection:
            result = connection.execute(query).first()
            if result is None or result.sd is None:
                return []
        buckets = Buckets.make_buckets(
            engine,
            column.table,
            SecondsDifference(column, anchor),
            join_tables,
        )
        return [
            DateAfterProposer(
                self._metadata,
                result.sd,
                result.mean,
                column,
                anchor,
                dialect=engine.dialect,
                buckets=buckets,
            )
        ]

    def get_proposers(
        self,
        columns: list[Column],
        engine: Engine,
    ) -> Sequence[Proposer]:
        """Get all proposers of dates that might be anchored to another column."""
        if len(columns) != 1:
            return []
        column = columns[0]
        ct = get_column_type(column)
        if not isinstance(ct, (Date, DateTime)):
            return []
        roles = _get_roles(self._config, column)
        if "start" not in roles:
            return []
        other_start_columns = [
            fk_col for fk_col in roles["start"] if fk_col[1] != column
        ]
        return [
            prop
            for anchor in other_start_columns
            for prop in self.make_date_after_proposers(engine, column, anchor[1])
        ]
