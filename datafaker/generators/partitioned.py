"""Powerful generators for numbers, choices and related missingness."""

from dataclasses import dataclass
from itertools import chain, combinations
from typing import Any, Iterable, Sequence, Union

import sqlalchemy
from sqlalchemy import Column, Connection, Engine, RowMapping, text
from sqlalchemy.types import Integer, Numeric

from datafaker.generators.base import Generator, dist_gen, get_column_type
from datafaker.generators.continuous import (
    CovariateQuery,
    MultivariateNormalGeneratorFactory,
)
from datafaker.utils import T, logger

NumericType = Union[int, float]

# How many distinct values can we have before we consider a
# choice distribution to be infeasible?
MAXIMUM_CHOICES = 500


def text_list(items: Iterable[str]) -> str:
    """Concatenate the items with commas and one "and"."""
    item_i = iter(items)
    try:
        last_item = next(item_i)
    except StopIteration:
        return ""
    try:
        so_far = next(item_i)
    except StopIteration:
        return last_item
    for item in item_i:
        so_far += ", " + last_item
        last_item = item
    return so_far + " and " + last_item


@dataclass
class RowPartition:
    """A partition where all the rows have the same pattern of NULLs."""

    query: str
    # list of numeric columns
    included_numeric: list[Column]
    # map of indices to column names that are being grouped by.
    # The indices are indices of where they need to be inserted into
    # the generator outputs.
    included_choice: dict[int, str]
    # map of column names to clause that defines the partition
    # such as "mycolumn IS NULL"
    excluded_columns: dict[str, str]
    # map of constant outputs that need to be inserted into the
    # list of included column values (so once the generator has
    # been run and the included_choice values have been
    # added): {index: value}
    constant_outputs: dict[int, Any]
    # The actual covariates from the source database
    covariates: Sequence[RowMapping]

    def comment(self) -> str:
        """Make an appropriate comment for this partition."""
        caveat = ""
        if self.included_choice:
            caveat = f" (for each possible value of {text_list(self.included_choice.values())})"
        if not self.included_numeric:
            return f"Number of rows for which {text_list(self.excluded_columns.values())}{caveat}"
        if not self.excluded_columns:
            where = ""
        else:
            where = f" where {text_list(self.excluded_columns.values())}"
        if len(self.included_numeric) == 1:
            return (
                f"Mean and variance for column {self.included_numeric[0].name}{where}."
            )
        return (
            "Means and covariate matrix for the columns "
            f"{text_list(col.name for col in self.included_numeric)}{where}{caveat} so that we can"
            " produce the relatedness between these in the fake data."
        )


@dataclass
class NullableColumn:
    """A reference to a nullable column whose nullability is part of a partitioning."""

    column: Column
    # The bit (power of two) of the number of the partition in the partition sizes list
    bitmask: int


class PartitionCountQuery:
    """Query, result and comment for the row counts of the null pattern partitions."""

    def __init__(
        self,
        connection: Connection,
        query: str,
        table_name: str,
        nullable_columns: Iterable[NullableColumn],
    ) -> None:
        """
        Initialise the partition count query.

        :param connection: Database connection.
        :param query: The query getting the row counts of the null pattern partitions.
        :param table_name: The name of the table being queried.
        :param nullable_columns: The columns that are being checked for nullness.
        """
        self.query = query
        rows = connection.execute(text(query)).mappings().fetchall()
        self.results = [dict(row) for row in rows]
        self.comment = (
            "Number of rows for each combination of the columns"
            f" { {nc.column.name for nc in nullable_columns} }"
            f" of the table {table_name} being null"
        )


class NullPartitionedNormalGenerator(Generator):
    """
    A generator of mixed numeric and non-numeric data.

    Generates data that matches the source data in
    missingness, choice of non-numeric data and numeric
    data.

    For the numeric data to be generated, samples of rows for each
    combination of non-numeric values and missingness. If any such
    combination has only one line in the source data (or sample of
    the source data if sampling), it will not be generated as a
    covariate matrix cannot be generated from one source row
    (although if the data is all non-numeric values and nulls, single
    rows are used because no covariate matrix is required for this).
    """

    # pylint: disable=too-many-arguments too-many-positional-arguments
    def __init__(
        self,
        query_name: str,
        partitions: dict[int, RowPartition],
        function_name: str = "grouped_multivariate_lognormal",
        name_suffix: str | None = None,
        partition_count_query: PartitionCountQuery | None = None,
    ):
        """Initialise a NullPartitionedNormalGenerator."""
        self._query_name = query_name
        self._partitions = partitions
        self._function_name = function_name
        self._partition_count_query = partition_count_query
        if name_suffix:
            self._name = f"null-partitioned {function_name} [{name_suffix}]"
        else:
            self._name = f"null-partitioned {function_name}"

    def name(self) -> str:
        """Get the name of the generator."""
        return self._name

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "dist_gen.alternatives"

    def _nominal_kwargs_with_combinations(
        self, index: int, partition: RowPartition
    ) -> dict[str, Any]:
        """Get the arguments to be entered into ``config.yaml`` for a single partition."""
        count = (
            'sum(r["count"] for r in'
            f' SRC_STATS["auto__cov__{self._query_name}__alt_{index}"]["results"])'
        )
        if not partition.included_numeric and not partition.included_choice:
            return {
                "count": count,
                "name": '"constant"',
                "params": {"value": [None] * len(partition.constant_outputs)},
            }
        covariates = {
            "covs": f'SRC_STATS["auto__cov__{self._query_name}__alt_{index}"]["results"]'
        }
        if not partition.constant_outputs:
            return {
                "count": count,
                "name": f'"{self._function_name}"',
                "params": covariates,
            }
        return {
            "count": count,
            "name": '"with_constants_at"',
            "params": {
                "constants_at": partition.constant_outputs,
                "subgen": f'"{self._function_name}"',
                "params": covariates,
            },
        }

    def _count_query_name(self) -> str:
        return f"auto__cov__{self._query_name}__counts"

    def nominal_kwargs(self) -> dict[str, Any]:
        """Get the arguments to be entered into ``config.yaml``."""
        return {
            "alternative_configs": [
                self._nominal_kwargs_with_combinations(index, self._partitions[index])
                for index in range(len(self._partitions))
            ],
            "counts": f'SRC_STATS["{self._count_query_name()}"]["results"]',
        }

    def custom_queries(self) -> dict[str, Any]:
        """Get the queries the generators need to call."""
        partitions = {
            f"auto__cov__{self._query_name}__alt_{index}": {
                "comment": partition.comment(),
                "query": partition.query,
            }
            for index, partition in self._partitions.items()
        }
        if not self._partition_count_query:
            return partitions
        return {
            self._count_query_name(): {
                "comment": self._partition_count_query.comment,
                "query": self._partition_count_query.query,
            },
            **partitions,
        }

    def _actual_kwargs_with_combinations(
        self, partition: RowPartition
    ) -> dict[str, Any]:
        count = sum(row["count"] for row in partition.covariates)
        if not partition.included_numeric and not partition.included_choice:
            return {
                "count": count,
                "name": "constant",
                "params": {"value": [None] * len(partition.excluded_columns)},
            }
        covariates = {
            "covs": partition.covariates,
        }
        if not partition.constant_outputs:
            return {
                "count": count,
                "name": self._function_name,
                "params": covariates,
            }
        return {
            "count": count,
            "name": "with_constants_at",
            "params": {
                "constants_at": partition.constant_outputs,
                "subgen": self._function_name,
                "params": covariates,
            },
        }

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        if self._partition_count_query is None:
            counts = None
        else:
            counts = self._partition_count_query.results
        return {
            "alternative_configs": [
                self._actual_kwargs_with_combinations(self._partitions[index])
                for index in range(len(self._partitions))
            ],
            "counts": counts,
        }

    def generate_data(self, count: int) -> list[Any]:
        """Generate 'count' random data points for this column."""
        kwargs = self.actual_kwargs()
        return [dist_gen.alternatives(**kwargs) for _ in range(count)]

    def fit(self, default: float = -1) -> float:
        """Get this generator's fit against the real data."""
        return default


def is_numeric(col: Column) -> bool:
    """Test if this column stores a numeric value."""
    ct = get_column_type(col)
    return isinstance(ct, (Numeric, Integer)) and not col.foreign_keys


def powerset(xs: list[T]) -> Iterable[Iterable[T]]:
    """Get a list of all sublists of ``input``."""
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


# pylint: disable=too-many-instance-attributes
class NullPatternPartition:
    """Get the definition of a partition (in other words, what makes it not another partition)."""

    def __init__(
        self, columns: Iterable[Column], partition_nonnulls: Iterable[NullableColumn]
    ):
        """Initialise a pattern of nulls which can be queried for."""
        self.index = sum(nc.bitmask for nc in partition_nonnulls)
        nonnull_columns = {nc.column.name for nc in partition_nonnulls}
        self.included_numeric: list[Column] = []
        self.included_choice: dict[int, str] = {}
        self.group_by_clause = ""
        self.constant_clauses = ""
        self.constants = ""
        self.excluded: dict[str, str] = {}
        self.predicates: list[str] = []
        self.nones: dict[int, None] = {}
        for col_index, column in enumerate(columns):
            col_name = column.name
            if col_name in nonnull_columns or not column.nullable:
                if is_numeric(column):
                    self.included_numeric.append(column)
                else:
                    index = len(self.included_numeric) + len(self.included_choice)
                    self.included_choice[index] = col_name
                    if self.group_by_clause:
                        self.group_by_clause += ", " + col_name
                    else:
                        self.group_by_clause = " GROUP BY " + col_name
                    self.constant_clauses += f", _q.{col_name} AS k{index}"
                    self.constants += ", " + col_name
            else:
                self.excluded[col_name] = f"{col_name} IS NULL"
                self.predicates.append(f"{col_name} IS NULL")
                self.nones[col_index] = None


class NullPartitionedNormalGeneratorFactory(MultivariateNormalGeneratorFactory):
    """Produces null partitioned generators, for complex interdependent data."""

    SAMPLE_COUNT = MAXIMUM_CHOICES
    SUPPRESS_COUNT = 7
    EMPTY_RESULT = [
        RowMapping(
            parent=sqlalchemy.engine.result.SimpleResultMetaData(["count"]),
            processors=None,
            key_to_index={"count": 0},
            data=(0,),
        )
    ]

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "grouped_multivariate_normal"

    def query_predicate(self, column: Column) -> str:
        """Get a SQL expression that is true when ``column`` is available for analysis."""
        if is_numeric(column):
            # x <> x + 1 ensures that x is not infinity or NaN
            return f"COALESCE({column.name} <> {column.name} + 1, FALSE)"
        return f"{column.name} IS NOT NULL"

    def query_var(self, column: str) -> str:
        """Return the expression we are querying for in this column."""
        return column

    def get_nullable_columns(self, columns: list[Column]) -> list[NullableColumn]:
        """Get a list of nullable columns together with bitmasks."""
        out: list[NullableColumn] = []
        for col in columns:
            if col.nullable:
                out.append(
                    NullableColumn(
                        column=col,
                        bitmask=2 ** len(out),
                    )
                )
        return out

    def get_partition_count_query(
        self, ncs: list[NullableColumn], table: str, where: str | None = None
    ) -> str:
        """
        Get a SQL expression returning columns ``count`` and ``index``.

        Each row returned represents one of the null pattern partitions.
        ``index`` is the bitmask of all those nullable columns that are not null for
        this partition, and ``count`` is the total number of rows in this partition.
        """
        index_exp = " + ".join(
            f"CASE WHEN {self.query_predicate(nc.column)} THEN {nc.bitmask} ELSE 0 END"
            for nc in ncs
        )
        if where is None:
            return f'SELECT COUNT(*) AS count, {index_exp} AS "index" FROM {table} GROUP BY "index"'
        return (
            'SELECT count, "index" FROM (SELECT COUNT(*) AS count,'
            f' {index_exp} AS "index"'
            f' FROM {table} GROUP BY "index") AS _q {where}'
        )

    # pylint: disable=too-many-arguments too-many-positional-arguments
    def _get_generator(
        self,
        connection: Connection,
        cov_query: CovariateQuery,
        columns: list[Column],
        nullable_columns: list[NullableColumn],
        name_suffix: str | None = None,
    ) -> NullPartitionedNormalGenerator | None:
        where = ""
        if 1 < cov_query.suppress_count:
            where = f' WHERE {cov_query.suppress_count} < "count"'
        query = self.get_partition_count_query(nullable_columns, cov_query.table, where)
        partitions: dict[int, RowPartition] = {}
        for partition_nonnulls in powerset(nullable_columns):
            partition_def = NullPatternPartition(columns, partition_nonnulls)
            cov_query.columns(
                partition_def.included_numeric,
            ).predicates(
                partition_def.predicates,
            ).group_by(
                partition_def.group_by_clause,
            ).constants(
                partition_def.constants,
            ).constant_clauses(
                partition_def.constant_clauses,
            )
            partitions[partition_def.index] = RowPartition(
                cov_query.get(),
                partition_def.included_numeric,
                partition_def.included_choice,
                partition_def.excluded,
                partition_def.nones,
                [],
            )
        if not self._execute_partition_queries(connection, partitions):
            return None
        return NullPartitionedNormalGenerator(
            f"{cov_query.table}__{columns[0].name}",
            partitions,
            self.function_name(),
            name_suffix=name_suffix,
            partition_count_query=PartitionCountQuery(
                connection,
                query,
                cov_query.table,
                nullable_columns,
            ),
        )

    def get_generators(
        self, columns: list[Column], engine: Engine
    ) -> Sequence[Generator]:
        """Get any appropriate generators for these columns."""
        if len(columns) < 2:
            return []
        nullable_columns = self.get_nullable_columns(columns)
        if not nullable_columns:
            return []
        table = columns[0].table.name
        gens: list[Generator | None] = []
        try:
            with engine.connect() as connection:
                cov_query = CovariateQuery(table, self)
                gens.append(
                    self._get_generator(
                        connection,
                        cov_query,
                        columns,
                        nullable_columns,
                    )
                )
                cov_query = cov_query.sample_count(self.SAMPLE_COUNT)
                gens.append(
                    self._get_generator(
                        connection,
                        cov_query,
                        columns,
                        nullable_columns,
                        name_suffix="sampled",
                    )
                )
                cov_query = CovariateQuery(table, self).set_suppress_count(
                    self.SUPPRESS_COUNT
                )
                gens.append(
                    self._get_generator(
                        connection,
                        cov_query,
                        columns,
                        nullable_columns,
                        name_suffix="suppressed",
                    )
                )
                cov_query = cov_query.sample_count(self.SAMPLE_COUNT)
                gens.append(
                    self._get_generator(
                        connection,
                        cov_query,
                        columns,
                        nullable_columns,
                        name_suffix="sampled and suppressed",
                    )
                )
        except sqlalchemy.exc.DatabaseError as exc:
            logger.debug("SQL query failed with error %s [%s]", exc, exc.statement)
            return []
        return [gen for gen in gens if gen]

    def _execute_partition_queries(
        self,
        connection: Connection,
        partitions: dict[int, RowPartition],
    ) -> bool:
        """
        Execute the query in each partition, filling in the covariates.

        :return: True if all the partitions work, False if any of them fail.
        """
        found_nonzero = False
        for rp in partitions.values():
            covs = connection.execute(text(rp.query)).mappings().fetchall()
            if not covs or covs.count == 0 or covs[0]["count"] is None:
                rp.covariates = self.EMPTY_RESULT
            else:
                rp.covariates = covs
                found_nonzero = True
        return found_nonzero


class NullPartitionedLogNormalGeneratorFactory(NullPartitionedNormalGeneratorFactory):
    """
    A generator for numeric and non-numeric columns.

    Any values could be null, the distributions of the nonnull numeric columns
    depend on each other and the other non-numeric column values.
    """

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "grouped_multivariate_lognormal"

    def query_predicate(self, column: Column) -> str:
        """Get the SQL expression testing if the value in this column should be used."""
        if is_numeric(column):
            # x <> x + 1 ensures that x is not infinity or NaN
            return f"COALESCE({column.name} <> {column.name} + 1 AND 0 < {column.name}, FALSE)"
        return f"{column.name} IS NOT NULL"

    def query_var(self, column: str) -> str:
        """Get the variable or expression we are querying for this column."""
        return f"LN({column})"
