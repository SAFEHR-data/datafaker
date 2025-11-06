"""Generator factories for making generators of continuous distributions."""

import itertools
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import Column, Engine, RowMapping, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import Integer, Numeric

from datafaker.generators.base import (
    Buckets,
    Generator,
    GeneratorFactory,
    NumericType,
    dist_gen,
    get_column_type,
)
from datafaker.utils import Empty, logger


class ContinuousDistributionGenerator(Generator):
    """Base class for generators producing continuous distributions."""

    expected_buckets: Sequence[NumericType] = []

    def __init__(self, table_name: str, column_name: str, buckets: Buckets):
        """Initialise a ContinuousDistributionGenerator."""
        super().__init__()
        self.table_name = table_name
        self.column_name = column_name
        self.buckets = buckets

    def nominal_kwargs(self) -> dict[str, Any]:
        """Get the arguments to be entered into ``config.yaml``."""
        return {
            "mean": (
                f'SRC_STATS["auto__{self.table_name}"]["results"]'
                f'[0]["mean__{self.column_name}"]'
            ),
            "sd": (
                f'SRC_STATS["auto__{self.table_name}"]["results"]'
                f'[0]["stddev__{self.column_name}"]'
            ),
        }

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        if self.buckets is None:
            return {}
        return {
            "mean": self.buckets.mean,
            "sd": self.buckets.stddev,
        }

    def select_aggregate_clauses(self) -> dict[str, dict[str, str]]:
        """Get the query fragments the generators need to call."""
        clauses = super().select_aggregate_clauses()
        return {
            **clauses,
            f"mean__{self.column_name}": {
                "clause": f"AVG({self.column_name})",
                "comment": f"Mean of {self.column_name} from table {self.table_name}",
            },
            f"stddev__{self.column_name}": {
                "clause": f"STDDEV({self.column_name})",
                "comment": f"Standard deviation of {self.column_name} from table {self.table_name}",
            },
        }

    def fit(self, default: float = -1) -> float:
        """Get this generator's fit against the real data."""
        if self.buckets is None:
            return default
        return self.buckets.fit_from_counts(self.expected_buckets)


class GaussianGenerator(ContinuousDistributionGenerator):
    """Generator producing numbers in a Gaussian (normal) distribution."""

    expected_buckets = [
        0.0227,
        0.0441,
        0.0918,
        0.1499,
        0.1915,
        0.1915,
        0.1499,
        0.0918,
        0.0441,
        0.0227,
    ]

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "dist_gen.normal"

    def generate_data(self, count: int) -> list[Any]:
        """Generate ``count`` random data points for this column."""
        return [
            dist_gen.normal(self.buckets.mean, self.buckets.stddev)
            for _ in range(count)
        ]


class UniformGenerator(ContinuousDistributionGenerator):
    """Generator producing numbers in a uniform distribution."""

    expected_buckets = [
        0,
        0.06698,
        0.14434,
        0.14434,
        0.14434,
        0.14434,
        0.14434,
        0.14434,
        0.06698,
        0,
    ]

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "dist_gen.uniform_ms"

    def generate_data(self, count: int) -> list[Any]:
        """Generate ``count`` random data points for this column."""
        return [
            dist_gen.uniform_ms(self.buckets.mean, self.buckets.stddev)
            for _ in range(count)
        ]


class ContinuousDistributionGeneratorFactory(GeneratorFactory):
    """All generators that want an average and standard deviation."""

    def _get_generators_from_buckets(
        self,
        _engine: Engine,
        table_name: str,
        column_name: str,
        buckets: Buckets,
    ) -> Sequence[Generator]:
        return [
            GaussianGenerator(table_name, column_name, buckets),
            UniformGenerator(table_name, column_name, buckets),
        ]

    def get_generators(
        self, columns: list[Column], engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators appropriate to these columns."""
        if len(columns) != 1:
            return []
        column = columns[0]
        ct = get_column_type(column)
        if not isinstance(ct, Numeric) and not isinstance(ct, Integer):
            return []
        column_name = column.name
        table_name = column.table.name
        buckets = Buckets.make_buckets(engine, table_name, column_name)
        if buckets is None:
            return []
        return self._get_generators_from_buckets(
            engine, table_name, column_name, buckets
        )


class LogNormalGenerator(Generator):
    """Generator producing numbers in a log-normal distribution."""

    # R:
    # > xs<-seq(-2,2,0.5)*sqrt((exp(1)-1)*exp(1))+exp(0.5)
    # > ys <- plnorm(xs)
    # > c(ys, 1) - c(0,ys)
    #  [1] 0.00000000 0.00000000 0.00000000 0.28589471 0.40556775 0.15086088
    #  [7] 0.06716451 0.03428958 0.01924848 0.03697409
    expected_buckets = [
        0,
        0,
        0,
        0.28589471,
        0.40556775,
        0.15086088,
        0.06716451,
        0.03428958,
        0.01924848,
        0.03697409,
    ]

    # pylint: disable=too-many-arguments too-many-positional-arguments
    def __init__(
        self,
        table_name: str,
        column_name: str,
        buckets: Buckets,
        logmean: float,
        logstddev: float,
    ):
        """Initialise a LogNormalGenerator."""
        super().__init__()
        self.table_name = table_name
        self.column_name = column_name
        self.buckets = buckets
        self.logmean = logmean
        self.logstddev = logstddev

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "dist_gen.lognormal"

    def generate_data(self, count: int) -> list[Any]:
        """Generate ``count`` random data points for this column."""
        return [dist_gen.lognormal(self.logmean, self.logstddev) for _ in range(count)]

    def nominal_kwargs(self) -> dict[str, Any]:
        """Get the arguments to be entered into ``config.yaml``."""
        return {
            "logmean": (
                f'SRC_STATS["auto__{self.table_name}"]["results"][0]'
                f'["logmean__{self.column_name}"]'
            ),
            "logsd": (
                f'SRC_STATS["auto__{self.table_name}"]["results"][0]'
                f'["logstddev__{self.column_name}"]'
            ),
        }

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        return {
            "logmean": self.logmean,
            "logsd": self.logstddev,
        }

    def select_aggregate_clauses(self) -> dict[str, dict[str, str]]:
        """Get the query fragments the generators need to call."""
        clauses = super().select_aggregate_clauses()
        return {
            **clauses,
            f"logmean__{self.column_name}": {
                "clause": (
                    f"AVG(CASE WHEN 0<{self.column_name} THEN LN({self.column_name})"
                    " ELSE NULL END)"
                ),
                "comment": f"Mean of logs of {self.column_name} from table {self.table_name}",
            },
            f"logstddev__{self.column_name}": {
                "clause": (
                    f"STDDEV(CASE WHEN 0<{self.column_name}"
                    f" THEN LN({self.column_name}) ELSE NULL END)"
                ),
                "comment": (
                    f"Standard deviation of logs of {self.column_name}"
                    f" from table {self.table_name}"
                ),
            },
        }

    def fit(self, default: float = -1) -> float:
        """Get this generator's fit against the real data."""
        if self.buckets is None:
            return default
        return self.buckets.fit_from_counts(self.expected_buckets)


class ContinuousLogDistributionGeneratorFactory(ContinuousDistributionGeneratorFactory):
    """All generators that want an average and standard deviation of log data."""

    def _get_generators_from_buckets(
        self,
        engine: Engine,
        table_name: str,
        column_name: str,
        buckets: Buckets,
    ) -> Sequence[Generator]:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    f"SELECT AVG(CASE WHEN 0<{column_name} THEN LN({column_name})"
                    " ELSE NULL END) AS logmean,"
                    f" STDDEV(CASE WHEN 0<{column_name} THEN LN({column_name}) ELSE NULL END)"
                    f" AS logstddev FROM {table_name}"
                )
            ).first()
            if result is None or result.logstddev is None:
                return []
        return [
            LogNormalGenerator(
                table_name,
                column_name,
                buckets,
                float(result.logmean),
                float(result.logstddev),
            )
        ]


class MultivariateNormalGenerator(Generator):
    """Generator of multiple values drawn from a multivariate normal distribution."""

    # pylint: disable=too-many-arguments too-many-positional-arguments
    def __init__(
        self,
        table_name: str,
        column_names: list[str],
        query: str,
        covariates: RowMapping,
        function_name: str,
    ) -> None:
        """Initialise a MultivariateNormalGenerator."""
        self._table = table_name
        self._columns = column_names
        self._query = query
        self._covariates = covariates
        self._function_name = function_name

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "dist_gen." + self._function_name

    def nominal_kwargs(self) -> dict[str, Any]:
        """Get the arguments to be entered into ``config.yaml``."""
        return {
            "cov": f'SRC_STATS["auto__cov__{self._table}"]["results"][0]',
        }

    def custom_queries(self) -> dict[str, Any]:
        """Get the queries the generators need to call."""
        cols = ", ".join(self._columns)
        return {
            f"auto__cov__{self._table}": {
                "comment": (
                    f"Means and covariate matrix for the columns {cols},"
                    " so that we can produce the relatedness between these in the fake data."
                ),
                "query": self._query,
            }
        }

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        return {"cov": self._covariates}

    def generate_data(self, count: int) -> list[Any]:
        """Generate 'count' random data points for this column."""
        return [
            getattr(dist_gen, self._function_name)(self._covariates)
            for _ in range(count)
        ]

    def fit(self, default: float = -1) -> float:
        """Get this generator's fit against the real data."""
        return default


class MultivariateNormalGeneratorFactory(GeneratorFactory):
    """Normal distribution generator factory."""

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "multivariate_normal"

    def query_predicate(self, column: Column) -> str:
        """Get the SQL expression for whether this column should be queried."""
        return column.name + " IS NOT NULL"

    def query_var(self, column: str) -> str:
        """Get the SQL expression of the value to query for this column."""
        return column

    # pylint: disable=too-many-arguments too-many-positional-arguments
    def query(
        self,
        table: str,
        columns: Sequence[Column],
        predicates: Iterable[str] = Empty.iterable(),
        group_by_clause: str = "",
        constant_clauses: str = "",
        constants: str = "",
        suppress_count: int = 1,
        sample_count: int | None = None,
    ) -> str:
        """
        Get a query for the basics for multivariate normal/lognormal parameters.

        :param table: The name of the table to be queried.
        :param columns: The columns in the multivariate distribution.
        :param and_where: Additional where clause. If not ``""`` should begin with ``" AND "``.
        :param group_by_clause: Any GROUP BY clause (starting with " GROUP BY " if not "").
        :param constant_clauses: Extra output columns in the outer SELECT clause, such
        as ", _q.column_one AS k1, _q.column_two AS k2". Note the initial comma.
        :param constants: Extra output columns in the inner SELECT clause. Used to
        deliver columns to the outer select, such as ", column_one, column_two".
        Note the initial comma.
        :param suppress_count: a group smaller than this will be suppressed.
        :param sample_count: this many samples will be taken from each partition.
        """
        means = "".join(f", _q.m{i}" for i in range(len(columns)))
        covs = "".join(
            (
                f", (_q.s{ix}_{iy} - _q.count * _q.m{ix} * _q.m{iy})"
                f"/NULLIF(_q.count - 1, 0) AS c{ix}_{iy}"
            )
            for iy in range(len(columns))
            for ix in range(iy + 1)
        )
        subquery = self._inner_query(table, columns, predicates, sample_count)
        # if there are any numeric columns we need at least
        # two rows to make any (co)variances at all
        suppress_clause = f" WHERE {suppress_count} < _q.count" if columns else ""
        return (
            f"SELECT {len(columns)} AS rank{constant_clauses}, _q.count AS count{means}{covs}"
            f" FROM ({self._middle_query(columns, constants, subquery, group_by_clause)})"
            f" AS _q{suppress_clause}"
        )

    def _inner_query(
        self,
        table: str,
        columns: Sequence[Column],
        predicates: Iterable[str],
        sample_count: int | None,
    ) -> str:
        """Get the rows from the table that we are interested in."""
        preds = itertools.chain(
            (self.query_predicate(col) for col in columns),
            predicates,
        )
        where = " AND ".join(preds) if preds else ""
        if where:
            where = " WHERE " + where
        if sample_count is None:
            return table + where
        return (
            f"(SELECT * FROM {table}{where} ORDER BY RANDOM()"
            f" LIMIT {sample_count}) AS _sampled"
        )

    def _middle_query(
        self,
        columns: Sequence[Column],
        constants: str,
        inner_query: str,
        group_by_clause: str,
    ) -> str:
        """Get the basic statistics (and constants) from the inner query."""
        multiples = "".join(
            f", SUM({self.query_var(colx.name)} * {self.query_var(coly.name)}) AS s{ix}_{iy}"
            for iy, coly in enumerate(columns)
            for ix, colx in enumerate(columns[: iy + 1])
        )
        avgs = "".join(
            f", AVG({self.query_var(col.name)}) AS m{i}"
            for i, col in enumerate(columns)
        )
        return (
            f"SELECT COUNT(*) AS count{multiples}{avgs}{constants}"
            f" FROM {inner_query}{group_by_clause}"
        )

    def get_generators(
        self, columns: list[Column], engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators for these columns."""
        # For the case of one column we'll use GaussianGenerator
        if len(columns) < 2:
            return []
        # All columns must be numeric
        for c in columns:
            ct = get_column_type(c)
            if not isinstance(ct, Numeric) and not isinstance(ct, Integer):
                return []
        column_names = [c.name for c in columns]
        table = columns[0].table.name
        query = self.query(table, columns)
        with engine.connect() as connection:
            try:
                covariates = connection.execute(text(query)).mappings().first()
            except SQLAlchemyError as e:
                logger.debug("SQL query %s failed with error %s", query, e)
                return []
            if not covariates or covariates["c0_0"] is None:
                return []
            return [
                MultivariateNormalGenerator(
                    table,
                    column_names,
                    query,
                    covariates,
                    self.function_name(),
                )
            ]


class MultivariateLogNormalGeneratorFactory(MultivariateNormalGeneratorFactory):
    """Multivariate lognormal generator factory."""

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "multivariate_lognormal"

    def query_predicate(self, column: Column) -> str:
        """Get the SQL expression for whether this column should be queried."""
        return f"COALESCE(0 < {column.name}, FALSE)"

    def query_var(self, column: str) -> str:
        """Get the expression to query for, for this column."""
        return f"LN({column})"
