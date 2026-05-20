"""Proposers for date intervals."""
import datetime
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import Column, Engine, ForeignKey, MetaData, func, select
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.elements import ColumnElement, coercions, roles
from sqlalchemy.types import Date, DateTime

from datafaker.proposers.base import Buckets, Proposer, ProposerFactory, get_column_type
from datafaker.providers import AnchoredProvider
from datafaker.utils import get_property

RelatedColumn = tuple[ForeignKey | None, Column]


class SecondsDifference(ColumnElement[int]):
    """Represent getting the difference between times in seconds."""

    expr1: ColumnElement[Date | DateTime]
    expr2: ColumnElement[Date | DateTime]

    def __init__(
        self,
        expr1: ColumnElement[Date | DateTime],
        expr2: ColumnElement[Date | DateTime],
    ):
        self.expr1 = expr1
        self.expr2 = expr2


@compiles(SecondsDifference)
def compile(element: SecondsDifference, compiler, **kw):
    e1 = compiler.process(element.expr1, **kw)
    e2 = compiler.process(element.expr2, **kw)
    return f"EXTRACT(EPOCH FROM ({e1})) - EXTRACT(EPOCH FROM ({e2}))"


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
    columns: list[Column],
) -> dict[str, list[RelatedColumn]]:
    """
    Work out where the roles are relative to this table.

    :param config: The configuration from ``config.yaml``.
    :param columns: The list of columns we are to propose for.
    :return: dictionary of ``role_name`` -> ``(fk or None, column_name)``
        where ``fk`` is the actual foreign key from the table, and ``None``
        means a column from the same table as the input column(s)
        has the required role.
    """
    if len(columns) == 0:
        return {}
    table = columns[0].table
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
        column_name: str,
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
        self._column_name = column_name
        self._provider = AnchoredProvider(metadata=metadata)
        if buckets is None:
            self._fit = None
            return
        dummy_anchor = datetime.datetime.fromisoformat("1970-01-01")
        samples = [(d - dummy_anchor).total_seconds() for d in self.generate_data(400)]
        self._fit = buckets.fit_from_values(samples)

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "anchored_provider.normal_date"

    def name(self) -> str:
        """Get the name of the generator."""
        return f"{self.function_name()} [anchored to {self._column_name}]"

    def nominal_kwargs(self) -> dict[str, Any]:
        """Get the arguments to be entered into ``config.yaml``."""
        return {
            "mean_seconds": self._sd,
            "sd_seconds": self._mean,
            "anchor": f'GENERATED_ROW["{self._column_name}"]',
        }

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        return {
            "mean_seconds": self._sd,
            "sd_seconds": self._mean,
            # For now we'll use a dummy value for the dependent value
            "anchor": "1970-01-01",
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
        with engine.connect() as connection:
            result = connection.execute(
                select(
                    func.avg(SecondsDifference(column, anchor)).label("mean"),
                    func.stddev(SecondsDifference(column, anchor)).label("sd"),
                ).select_from(column.table)
            ).first()
            if result is None or result.sd is None:
                return []
        buckets = Buckets.make_buckets(engine, column.table, column - anchor)
        return [
            DateAfterProposer(
                self._metadata,
                result.sd,
                result.mean,
                anchor.name,
                buckets,
            )
        ]

    def get_proposers(
        self,
        columns: list[Column],
        engine: Engine,
    ) -> Sequence[Proposer]:
        """Get ``DateAfterProposers`` suitable for this column."""
        if len(columns) != 1:
            return []
        column = columns[0]
        ct = get_column_type(column)
        if not isinstance(ct, (Date, DateTime)):
            return []
        roles = _get_roles(self._config, columns)
        if "start" not in roles:
            return []
        other_start_columns = [
            fk_col for fk_col in roles["start"] if fk_col[1] not in columns
        ]
        # For the moment, only anchors in _this_ table
        anchors = [
            anchor
            for fk, anchor in other_start_columns
            if fk is None  # at the moment we can only cope with columns in the same table
        ]
        return [
            prop
            for anchor in anchors
            for prop in self.make_date_after_proposers(engine, column, anchor)
        ]
