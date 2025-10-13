"""Basic Generators and factories."""

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Sequence, Union

import mimesis
import mimesis.locales
import sqlalchemy
from sqlalchemy import Column, Engine, text
from sqlalchemy.types import Integer, Numeric, String, TypeEngine
from typing_extensions import Self

from datafaker.base import DistributionGenerator
from datafaker.utils import T, logger

NumericType = Union[int, float]


dist_gen = DistributionGenerator()
generic = mimesis.Generic(locale=mimesis.locales.Locale.EN_GB)


class Generator(ABC):
    """
    Random data generator.

    A generator is specific to a particular column in a particular table in
    a particluar database.

    A generator knows how to fetch its summary data from the database, how to calculate
    its fit (if apropriate) and which function actually does the generation.

    It also knows these summary statistics for the column it was instantiated on,
    and therefore knows how to generate fake data for that column.
    """

    @abstractmethod
    def function_name(self) -> str:
        """Get the name of the generator function to put into df.py."""

    def name(self) -> str:
        """
        Get the name of the generator.

        Usually the same as the function name, but can be different to distinguish
        between generators that have the same function but different queries.
        """
        return self.function_name()

    @abstractmethod
    def nominal_kwargs(self) -> dict[str, str]:
        """
        Get the kwargs the generator wants to be called with.

        The values will tend to be references to something in the src-stats.yaml
        file.
        For example {"avg_age": 'SRC_STATS["auto__patient"]["results"][0]["age_mean"]'} will
        provide the value stored in src-stats.yaml as
        SRC_STATS["auto__patient"]["results"][0]["age_mean"] as the "avg_age" argument
        to the generator function.
        """

    def select_aggregate_clauses(self) -> dict[str, dict[str, str]]:
        """
        Get the SQL clauses to add to a SELECT ... FROM {table} query.

        Will add to SRC_STATS["auto__{table}"]
        For example {
            "count": {
                "clause": "COUNT(*)",
                "comment": "number of rows in table {table}"
            }, "avg_thiscolumn": {
                "clause": "AVG(thiscolumn)",
                "comment": "Average value of thiscolumn in table {table}"
        }}
        will make the clause become:
        "SELECT COUNT(*) AS count, AVG(thiscolumn) AS avg_thiscolumn FROM thistable"
        and this will populate SRC_STATS["auto__thistable"]["results"][0]["count"] and
        SRC_STATS["auto__thistable"]["results"][0]["avg_thiscolumn"] in the src-stats.yaml file.
        """
        return {}

    def custom_queries(self) -> dict[str, dict[str, str]]:
        """
        Get the SQL queries to add to SRC_STATS.

        Should be used for queries that do not follow the SELECT ... FROM table format
        using aggregate queries, because these should use select_aggregate_clauses.

        For example {"myquery": {
            "query": "SELECT one, too AS two FROM mytable WHERE too > 1",
            "comment": "big enough one and two from table mytable"
        }}
        will populate SRC_STATS["myquery"]["results"][0]["one"]
        and SRC_STATS["myquery"]["results"][0]["two"]
        in the src-stats.yaml file.

        Keys should be chosen to minimize the chances of clashing with other queries,
        for example "auto__{table}__{column}__{queryname}"
        """
        return {}

    @abstractmethod
    def actual_kwargs(self) -> dict[str, Any]:
        """
        Get the kwargs (summary statistics) this generator is instantiated with.

        This must match `nominal_kwargs` in structure.
        """

    @abstractmethod
    def generate_data(self, count: int) -> list[Any]:
        """Generate ``count`` random data points for this column."""

    def fit(self, default: float = -1) -> float:
        """
        Return a value representing how well the distribution fits the real source data.

        0.0 means "perfectly".
        Returns default if no fitness has been defined.
        """
        return default


class PredefinedGenerator(Generator):
    """Generator built from an existing config.yaml."""

    SELECT_AGGREGATE_RE = re.compile(r"SELECT (.*) FROM ([A-Za-z_][A-Za-z0-9_]*)")
    AS_CLAUSE_RE = re.compile(r" *(.+) +AS +([A-Za-z_][A-Za-z0-9_]*) *")
    SRC_STAT_NAME_RE = re.compile(r'\bSRC_STATS\["([^]]*)"\].*')

    def _get_src_stats_mentioned(self, val: Any) -> set[str]:
        if not val:
            return set()
        if isinstance(val, str):
            ss = self.SRC_STAT_NAME_RE.match(val)
            if ss:
                ss_name = ss.group(1)
                logger.debug("Found SRC_STATS reference %s", ss_name)
                return set([ss_name])
            logger.debug("Value %s does not seem to be a SRC_STATS reference", val)
            return set()
        if isinstance(val, list):
            return set.union(*(self._get_src_stats_mentioned(v) for v in val))
        if isinstance(val, dict):
            return set.union(*(self._get_src_stats_mentioned(v) for v in val.values()))
        return set()

    def __init__(
        self,
        table_name: str,
        generator_object: Mapping[str, Any],
        config: Mapping[str, Any],
    ):
        """
        Initialise a generator from a config.yaml.

        :param config: The entire configuration.
        :param generator_object: The part of the configuration at tables.*.row_generators
        """
        logger.debug(
            "Creating a PredefinedGenerator %s from table %s",
            generator_object["name"],
            table_name,
        )
        self._table_name = table_name
        self._name: str = generator_object["name"]
        self._kwn: dict[str, str] = generator_object.get("kwargs", {})
        self._src_stats_mentioned = self._get_src_stats_mentioned(self._kwn)
        # Need to deal with this somehow (or remove it from the schema)
        self._argn: list[str] = generator_object.get("args", [])
        self._select_aggregate_clauses: dict[str, dict[str, str | Any]] = {}
        self._custom_queries = {}
        for sstat in config.get("src-stats", []):
            name: str = sstat["name"]
            dpq = sstat.get("dp-query", None)
            query = sstat.get(
                "query", dpq
            )  # ... should we really be combining query and dp-query?
            comments = sstat.get("comments", [])
            if name in self._src_stats_mentioned:
                logger.debug("Found a src-stats entry for %s", name)
                # This query is one that this generator is interested in
                sam = None if query is None else self.SELECT_AGGREGATE_RE.match(query)
                # sam.group(2) is the table name from the FROM clause of the query
                if sam and name == f"auto__{sam.group(2)}":
                    # name is auto__{table_name}, so it's a select_aggregate,
                    # so we split up its clauses
                    sacs = [
                        self.AS_CLAUSE_RE.match(clause)
                        for clause in sam.group(1).split(",")
                    ]
                    # Work out what select_aggregate_clauses this represents
                    for sac in sacs:
                        if sac is not None:
                            comment = comments.pop() if comments else None
                            self._select_aggregate_clauses[sac.group(2)] = {
                                "clause": sac.group(1),
                                "comment": comment,
                            }
                else:
                    # some other name, so must be a custom query
                    logger.debug("Custom query %s is '%s'", name, query)
                    self._custom_queries[name] = {
                        "query": query,
                        "comment": comments[0] if comments else None,
                    }

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return self._name

    def nominal_kwargs(self) -> dict[str, str]:
        """Get the arguments to be entered into ``config.yaml``."""
        return self._kwn

    def select_aggregate_clauses(self) -> dict[str, dict[str, str]]:
        """Get the query fragments the generators need to call."""
        return self._select_aggregate_clauses

    def custom_queries(self) -> dict[str, dict[str, str]]:
        """Get the queries the generators need to call."""
        return self._custom_queries

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        # Run the queries from nominal_kwargs
        # ...
        logger.error("PredefinedGenerator.actual_kwargs not implemented yet")
        return {}

    def generate_data(self, count: int) -> list[Any]:
        """Generate ``count`` random data points for this column."""
        # Call the function if we can. This could be tricky...
        # ...
        logger.error("PredefinedGenerator.generate_data not implemented yet")
        return []


class GeneratorFactory(ABC):
    """A factory for making generators appropriate for a database column."""

    @abstractmethod
    def get_generators(
        self, columns: list[Column], engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators appropriate to these columns."""


def fit_from_buckets(xs: Sequence[NumericType], ys: Sequence[NumericType]) -> float:
    """Calculate the fit by comparing a pair of lists of buckets."""
    sum_diff_squared = sum(map(lambda t, a: (t - a) * (t - a), xs, ys))
    count = len(ys)
    return sum_diff_squared / (count * count)


class Buckets:
    """
    Measured buckets for a real distribution.

    Finds the real distribution of continuous data so that we can measure
    the fit of generators against it.
    """

    def __init__(
        self,
        engine: Engine,
        table_name: str,
        column_name: str,
        mean: float,
        stddev: float,
        count: int,
    ):
        """Initialise a Buckets object."""
        with engine.connect() as connection:
            raw_buckets = connection.execute(
                text(
                    f"SELECT COUNT({column_name}) AS f,"
                    f" FLOOR(({column_name} - {mean - 2 * stddev})/{stddev / 2}) AS b"
                    f" FROM {table_name} GROUP BY b"
                )
            )
            self.buckets: Sequence[int] = [0] * 10
            for rb in raw_buckets:
                if rb.b is not None:
                    bucket = min(9, max(0, int(rb.b) + 1))
                    self.buckets[bucket] += rb.f / count
            self.mean = mean
            self.stddev = stddev

    @classmethod
    def make_buckets(
        cls, engine: Engine, table_name: str, column_name: str
    ) -> Self | None:
        """
        Construct a Buckets object.

        Calculates the mean and standard deviation of the values in the column
        specified and makes ten buckets, centered on the mean and each half
        a standard deviation wide (except for the end two that extend to
        infinity). Each bucket will be set to the count of the number of values
        in the column within that bucket.
        """
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    f"SELECT AVG({column_name}) AS mean,"
                    f" STDDEV({column_name}) AS stddev,"
                    f" COUNT({column_name}) AS count FROM {table_name}"
                )
            ).first()
            if result is None or result.stddev is None or getattr(result, "count") < 2:
                return None
        try:
            buckets = cls(
                engine,
                table_name,
                column_name,
                result.mean,
                result.stddev,
                getattr(result, "count"),
            )
        except sqlalchemy.exc.DatabaseError as exc:
            logger.debug("Failed to instantiate Buckets object: %s", exc)
            return None
        return buckets

    def fit_from_counts(self, bucket_counts: Sequence[float]) -> float:
        """Figure out the fit from bucket counts from the generator distribution."""
        return fit_from_buckets(self.buckets, bucket_counts)

    def fit_from_values(self, values: list[float]) -> float:
        """Figure out the fit from samples from the generator distribution."""
        buckets = [0] * 10
        x = self.mean - 2 * self.stddev
        w = self.stddev / 2
        for v in values:
            b = min(9, max(0, int((v - x) / w)))
            buckets[b] += 1
        return self.fit_from_counts(buckets)


class MultiGeneratorFactory(GeneratorFactory):
    """A composite factory."""

    def __init__(self, factories: list[GeneratorFactory]):
        """Initialise a MultiGeneratorFactory."""
        super().__init__()
        self.factories = factories

    def get_generators(
        self, columns: list[Column], engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators appropriate to these columns."""
        return [
            generator
            for factory in self.factories
            for generator in factory.get_generators(columns, engine)
        ]


def get_column_type(column: Column) -> TypeEngine:
    """Get the type of the column, generic if possible."""
    try:
        return column.type.as_generic()
    except NotImplementedError:
        return column.type


class ConstantGenerator(Generator):
    """Generator that always produces the same value."""

    def __init__(self, value: Any) -> None:
        """Initialise the ConstantGenerator."""
        super().__init__()
        self.value = value
        self.repr = repr(value)

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "dist_gen.constant"

    def nominal_kwargs(self) -> dict[str, str]:
        """Get the arguments to be entered into ``config.yaml``."""
        return {"value": self.repr}

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        return {"value": self.value}

    def generate_data(self, count: int) -> list[Any]:
        """Generate ``count`` random data points for this column."""
        return [self.value for _ in range(count)]


class ConstantGeneratorFactory(GeneratorFactory):
    """Just the null generator."""

    def get_generators(
        self, columns: list[Column], engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators appropriate for these columns."""
        if len(columns) != 1:
            return []
        column = columns[0]
        if column.nullable:
            return [ConstantGenerator(None)]
        c_type = get_column_type(column)
        if isinstance(c_type, String):
            return [ConstantGenerator("")]
        if isinstance(c_type, Numeric):
            return [ConstantGenerator(0.0)]
        if isinstance(c_type, Integer):
            return [ConstantGenerator(0)]
        return []
