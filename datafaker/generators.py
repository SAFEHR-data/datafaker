"""
Generator factories for making generators for single columns.
"""

import decimal
import math
import re
import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain, combinations
from typing import Any, Callable, Iterable, Self, TypeVar

import mimesis
import mimesis.locales
import sqlalchemy
from sqlalchemy import Column, Connection, Engine, RowMapping, Sequence, text
from sqlalchemy.types import Date, DateTime, Integer, Numeric, String, Time

from datafaker.base import DistributionGenerator
from datafaker.utils import logger

# How many distinct values can we have before we consider a
# choice distribution to be infeasible?
MAXIMUM_CHOICES = 500

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
        """The name of the generator function to put into df.py."""

    def name(self) -> str:
        """
        The name of the generator.

        Usually the same as the function name, but can be different to distinguish
        between generators that have the same function but different queries.
        """
        return self.function_name()

    @abstractmethod
    def nominal_kwargs(self) -> dict[str, str]:
        """
        The kwargs the generator wants to be called with.
        The values will tend to be references to something in the src-stats.yaml
        file.
        For example {"avg_age": 'SRC_STATS["auto__patient"]["results"][0]["age_mean"]'} will
        provide the value stored in src-stats.yaml as
        SRC_STATS["auto__patient"]["results"][0]["age_mean"] as the "avg_age" argument
        to the generator function.
        """

    def select_aggregate_clauses(self) -> dict[str, dict[str, str]]:
        """
        SQL clauses to add to a SELECT ... FROM {table} query.

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
        SQL queries to add to SRC_STATS.

        Should be used for queries that do not follow the SELECT ... FROM table format
        using aggregate queries, because these should use select_aggregate_clauses.

        For example {"myquery": {
            "query": "SELECT one, too AS two FROM mytable WHERE too > 1",
            "comment": "big enough one and two from table mytable"
        }}
        will populate SRC_STATS["myquery"]["results"][0]["one"] and SRC_STATS["myquery"]["results"][0]["two"]
        in the src-stats.yaml file.

        Keys should be chosen to minimize the chances of clashing with other queries,
        for example "auto__{table}__{column}__{queryname}"
        """
        return {}

    @abstractmethod
    def actual_kwargs(self) -> dict[str, Any]:
        """
        The kwargs (summary statistics) this generator is instantiated with.
        """

    @abstractmethod
    def generate_data(self, count: int) -> list[Any]:
        """
        Generate 'count' random data points for this column.
        """

    def fit(self, default: float = None) -> float | None:
        """
        Return a value representing how well the distribution fits the real source data.

        0.0 means "perfectly".
        Returns default if no fitness has been defined.
        """
        return default


class PredefinedGenerator(Generator):
    """
    Generator built from an existing config.yaml.
    """

    SELECT_AGGREGATE_RE = re.compile(r"SELECT (.*) FROM ([A-Za-z_][A-Za-z0-9_]*)")
    AS_CLAUSE_RE = re.compile(r" *(.+) +AS +([A-Za-z_][A-Za-z0-9_]*) *")
    SRC_STAT_NAME_RE = re.compile(r'\bSRC_STATS\["([^]]*)"\].*')

    def _get_src_stats_mentioned(self, val: Any) -> set[str]:
        if not val:
            return set()
        if type(val) is str:
            ss = self.SRC_STAT_NAME_RE.match(val)
            if ss:
                ss_name = ss.group(1)
                logger.debug("Found SRC_STATS reference %s", ss_name)
                return set([ss_name])
            else:
                logger.debug("Value %s does not seem to be a SRC_STATS reference", val)
                return set()
        if type(val) is list:
            return set.union(*(self._get_src_stats_mentioned(v) for v in val))
        if type(val) is dict:
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
        self._select_aggregate_clauses = {}
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
                    # name is auto__{table_name}, so it's a select_aggregate, so we split up its clauses
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
        return self._name

    def nominal_kwargs(self) -> dict[str, str]:
        return self._kwn

    def select_aggregate_clauses(self) -> dict[str, dict[str, str]]:
        return self._select_aggregate_clauses

    def custom_queries(self) -> dict[str, dict[str, str]]:
        return self._custom_queries

    def actual_kwargs(self) -> dict[str, Any]:
        # Run the queries from nominal_kwargs
        # ...
        logger.error("PredefinedGenerator.actual_kwargs not implemented yet")
        return {}

    def generate_data(self, count: int) -> list[Any]:
        # Call the function if we can. This could be tricky...
        # ...
        logger.error("PredefinedGenerator.generate_data not implemented yet")
        return []


class GeneratorFactory(ABC):
    """
    A factory for making generators appropriate for a database column.
    """

    @abstractmethod
    def get_generators(self, columns: list[Column], engine: Engine) -> list[Generator]:
        """
        Returns all the generators that might be appropriate for this column.
        """


class Buckets:
    """
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
        with engine.connect() as connection:
            raw_buckets = connection.execute(
                text(
                    "SELECT COUNT({column}) AS f, FLOOR(({column} - {x})/{w}) AS b FROM {table} GROUP BY b".format(
                        column=column_name,
                        table=table_name,
                        x=mean - 2 * stddev,
                        w=stddev / 2,
                    )
                )
            )
            self.buckets = [0] * 10
            for rb in raw_buckets:
                if rb.b is not None:
                    bucket = min(9, max(0, int(rb.b) + 1))
                    self.buckets[bucket] += rb.f / count
            self.mean = mean
            self.stddev = stddev

    @classmethod
    def make_buckets(
        _cls, engine: Engine, table_name: str, column_name: str
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
                    "SELECT AVG({column}) AS mean, STDDEV({column}) AS stddev, COUNT({column}) AS count FROM {table}".format(
                        table=table_name,
                        column=column_name,
                    )
                )
            ).first()
            if result is None or result.stddev is None or result.count < 2:
                return None
        try:
            buckets = Buckets(
                engine,
                table_name,
                column_name,
                result.mean,
                result.stddev,
                result.count,
            )
        except sqlalchemy.exc.DatabaseError as exc:
            logger.debug("Failed to instantiate Buckets object: %s", exc)
            return None
        return buckets

    def fit_from_counts(self, bucket_counts: list[float]) -> float:
        """
        Figure out the fit from bucket counts from the generator distribution.
        """
        return fit_from_buckets(self.buckets, bucket_counts)

    def fit_from_values(self, values: list[float]) -> float:
        """
        Figure out the fit from samples from the generator distribution.
        """
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
        super().__init__()
        self.factories = factories

    def get_generators(self, columns: list[Column], engine: Engine) -> list[Generator]:
        return [
            generator
            for factory in self.factories
            for generator in factory.get_generators(columns, engine)
        ]


class MimesisGeneratorBase(Generator):
    def __init__(
        self,
        function_name: str,
    ):
        """
        Generator from Mimesis.

        :param function_name: is relative to 'generic', for example 'person.name'.
        """
        super().__init__()
        f = generic
        for part in function_name.split("."):
            if not hasattr(f, part):
                raise Exception(
                    f"Mimesis does not have a function {function_name}: {part} not found"
                )
            f = getattr(f, part)
        if not callable(f):
            raise Exception(
                f"Mimesis object {function_name} is not a callable, so cannot be used as a generator"
            )
        self._name = "generic." + function_name
        self._generator_function = f

    def function_name(self):
        return self._name

    def generate_data(self, count):
        return [self._generator_function() for _ in range(count)]


class MimesisGenerator(MimesisGeneratorBase):
    def __init__(
        self,
        function_name: str,
        value_fn: Callable[[Any], float] | None = None,
        buckets: Buckets | None = None,
    ):
        """
        Generator from Mimesis.

        :param function_name: is relative to 'generic', for example 'person.name'.
        :param value_fn: Function to convert generator output to floats, if needed. The values
        thus produced are compared against the buckets to estimate the fit.
        :param buckets: The distribution of string lengths in the real data. If this is None
        then the fit method will return None.
        """
        super().__init__(function_name)
        if buckets is None:
            self._fit = None
            return
        samples = self.generate_data(400)
        if value_fn:
            samples = [value_fn(s) for s in samples]
        self._fit = buckets.fit_from_values(samples)

    def function_name(self):
        return self._name

    def nominal_kwargs(self):
        return {}

    def actual_kwargs(self):
        return {}

    def fit(self, default=None):
        return default if self._fit is None else self._fit


class MimesisGeneratorTruncated(MimesisGenerator):
    def __init__(
        self,
        function_name: str,
        length: int,
        value_fn: Callable[[Any], float] | None = None,
        buckets: Buckets | None = None,
    ):
        self._length = length
        super().__init__(function_name, value_fn, buckets)

    def function_name(self) -> str:
        return "dist_gen.truncated_string"

    def name(self) -> str:
        return f"{self._name} [truncated to {self._length}]"

    def nominal_kwargs(self):
        return {
            "subgen_fn": self._name,
            "params": {},
            "length": self._length,
        }

    def actual_kwargs(self):
        return {
            "subgen_fn": self._name,
            "params": {},
            "length": self._length,
        }

    def generate_data(self, count):
        return [self._generator_function()[: self._length] for _ in range(count)]


class MimesisDateTimeGenerator(MimesisGeneratorBase):
    def __init__(
        self,
        column: Column,
        function_name: str,
        min_year: str,
        max_year: str,
        start: int,
        end: int,
    ):
        """
        :param column: The column to generate into
        :param function_name: The name of the mimesis function
        :param min_year: SQL expression extracting the minimum year
        :param min_year: SQL expression extracting the maximum year
        :param start: The actual first year found
        :param end: The actual last year found
        """
        super().__init__(function_name)
        self._column = column
        self._max_year = max_year
        self._min_year = min_year
        self._start = start
        self._end = end

    @classmethod
    def make_singleton(_cls, column: Column, engine: Engine, function_name: str):
        extract_year = f"CAST(EXTRACT(YEAR FROM {column.name}) AS INT)"
        max_year = f"MAX({extract_year})"
        min_year = f"MIN({extract_year})"
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    f"SELECT {min_year} AS start, {max_year} AS end FROM {column.table.name}"
                )
            ).first()
            if result is None or result.start is None or result.end is None:
                return []
        return [
            MimesisDateTimeGenerator(
                column,
                function_name,
                min_year,
                max_year,
                int(result.start),
                int(result.end),
            )
        ]

    def nominal_kwargs(self):
        return {
            "start": f'SRC_STATS["auto__{self._column.table.name}"]["results"][0]["{self._column.name}__start"]',
            "end": f'SRC_STATS["auto__{self._column.table.name}"]["results"][0]["{self._column.name}__end"]',
        }

    def actual_kwargs(self):
        return {
            "start": self._start,
            "end": self._end,
        }

    def select_aggregate_clauses(self) -> dict[str, dict[str, str]]:
        return {
            f"{self._column.name}__start": {
                "clause": self._min_year,
                "comment": f"Earliest year found for column {self._column.name} in table {self._column.table.name}",
            },
            f"{self._column.name}__end": {
                "clause": self._max_year,
                "comment": f"Latest year found for column {self._column.name} in table {self._column.table.name}",
            },
        }

    def generate_data(self, count):
        return [
            self._generator_function(start=self._start, end=self._end)
            for _ in range(count)
        ]


def get_column_type(column: Column):
    try:
        return column.type.as_generic()
    except NotImplementedError:
        return column.type


class MimesisStringGeneratorFactory(GeneratorFactory):
    """
    All Mimesis generators that return strings.
    """

    GENERATOR_NAMES = [
        "address.calling_code",
        "address.city",
        "address.continent",
        "address.country",
        "address.country_code",
        "address.postal_code",
        "address.province",
        "address.street_number",
        "address.street_name",
        "address.street_suffix",
        "person.blood_type",
        "person.email",
        "person.first_name",
        "person.last_name",
        "person.full_name",
        "person.gender",
        "person.language",
        "person.nationality",
        "person.occupation",
        "person.password",
        "person.title",
        "person.university",
        "person.username",
        "person.worldview",
        "text.answer",
        "text.color",
        "text.level",
        "text.quote",
        "text.sentence",
        "text.text",
        "text.word",
    ]

    def get_generators(self, columns: list[Column], engine: Engine):
        if len(columns) != 1:
            return []
        column = columns[0]
        column_type = get_column_type(column)
        if not isinstance(column_type, String):
            return []
        try:
            buckets = Buckets.make_buckets(
                engine,
                column.table.name,
                f"LENGTH({column.name})",
            )
            fitness_fn = len
        except Exception as exc:
            # Some column types that appear to be strings (such as enums)
            # cannot have their lengths measured. In this case we cannot
            # detect fitness using lengths.
            buckets = None
            fitness_fn = None
        length = column_type.length
        if length:
            return list(
                map(
                    lambda gen: MimesisGeneratorTruncated(
                        gen, length, fitness_fn, buckets
                    ),
                    self.GENERATOR_NAMES,
                )
            )
        return list(
            map(
                lambda gen: MimesisGenerator(gen, fitness_fn, buckets),
                self.GENERATOR_NAMES,
            )
        )


class MimesisFloatGeneratorFactory(GeneratorFactory):
    """
    All Mimesis generators that return floating point numbers.
    """

    def get_generators(self, columns: list[Column], engine: Engine):
        if len(columns) != 1:
            return []
        column = columns[0]
        if not isinstance(get_column_type(column), Numeric):
            return []
        return list(
            map(
                MimesisGenerator,
                [
                    "person.height",
                ],
            )
        )


class MimesisDateGeneratorFactory(GeneratorFactory):
    """
    All Mimesis generators that return dates.
    """

    def get_generators(self, columns: list[Column], engine: Engine):
        if len(columns) != 1:
            return []
        column = columns[0]
        ct = get_column_type(column)
        if not isinstance(ct, Date):
            return []
        return MimesisDateTimeGenerator.make_singleton(column, engine, "datetime.date")


class MimesisDateTimeGeneratorFactory(GeneratorFactory):
    """
    All Mimesis generators that return datetimes.
    """

    def get_generators(self, columns: list[Column], engine: Engine):
        if len(columns) != 1:
            return []
        column = columns[0]
        ct = get_column_type(column)
        if not isinstance(ct, DateTime):
            return []
        return MimesisDateTimeGenerator.make_singleton(
            column, engine, "datetime.datetime"
        )


class MimesisTimeGeneratorFactory(GeneratorFactory):
    """
    All Mimesis generators that return times.
    """

    def get_generators(self, columns: list[Column], engine: Engine):
        if len(columns) != 1:
            return []
        column = columns[0]
        ct = get_column_type(column)
        if not isinstance(ct, Time):
            return []
        return [MimesisGenerator("datetime.time")]


class MimesisIntegerGeneratorFactory(GeneratorFactory):
    """
    All Mimesis generators that return integers.
    """

    def get_generators(self, columns: list[Column], engine: Engine):
        if len(columns) != 1:
            return []
        column = columns[0]
        ct = get_column_type(column)
        if not isinstance(ct, Numeric) and not isinstance(ct, Integer):
            return []
        return [MimesisGenerator("person.weight")]


def fit_from_buckets(xs: list[float], ys: list[float]):
    sum_diff_squared = sum(map(lambda t, a: (t - a) * (t - a), xs, ys))
    count = len(ys)
    return sum_diff_squared / (count * count)


class ContinuousDistributionGenerator(Generator):
    def __init__(self, table_name: str, column_name: str, buckets: Buckets):
        super().__init__()
        self.table_name = table_name
        self.column_name = column_name
        self.buckets = buckets

    def nominal_kwargs(self):
        return {
            "mean": f'SRC_STATS["auto__{self.table_name}"]["results"][0]["mean__{self.column_name}"]',
            "sd": f'SRC_STATS["auto__{self.table_name}"]["results"][0]["stddev__{self.column_name}"]',
        }

    def actual_kwargs(self):
        if self.buckets is None:
            return {}
        return {
            "mean": self.buckets.mean,
            "sd": self.buckets.stddev,
        }

    def select_aggregate_clauses(self) -> dict[str, dict[str, str]]:
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

    def fit(self, default=None):
        if self.buckets is None:
            return default
        return self.buckets.fit_from_counts(self.expected_buckets)


class GaussianGenerator(ContinuousDistributionGenerator):
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

    def function_name(self):
        return "dist_gen.normal"

    def generate_data(self, count):
        return [
            dist_gen.normal(self.buckets.mean, self.buckets.stddev)
            for _ in range(count)
        ]


class UniformGenerator(ContinuousDistributionGenerator):
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

    def function_name(self):
        return "dist_gen.uniform_ms"

    def generate_data(self, count):
        return [
            dist_gen.uniform_ms(self.buckets.mean, self.buckets.stddev)
            for _ in range(count)
        ]


class ContinuousDistributionGeneratorFactory(GeneratorFactory):
    """
    All generators that want an average and standard deviation.
    """

    def _get_generators_from_buckets(
        self,
        _engine: Engine,
        table_name: str,
        column_name: str,
        buckets: Buckets,
    ) -> list[Generator]:
        return [
            GaussianGenerator(table_name, column_name, buckets),
            UniformGenerator(table_name, column_name, buckets),
        ]

    def get_generators(self, columns: list[Column], engine: Engine) -> list[Generator]:
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
    # TODO: figure out the real buckets here (this was from a random sample in R)
    expected_buckets = [
        0,
        0,
        0,
        0.28627,
        0.40607,
        0.14937,
        0.06735,
        0.03492,
        0.01918,
        0.03684,
    ]

    def __init__(
        self,
        table_name: str,
        column_name: str,
        buckets: Buckets,
        logmean: float,
        logstddev: float,
    ):
        super().__init__()
        self.table_name = table_name
        self.column_name = column_name
        self.buckets = buckets
        self.logmean = logmean
        self.logstddev = logstddev

    def function_name(self):
        return "dist_gen.lognormal"

    def generate_data(self, count):
        return [dist_gen.lognormal(self.logmean, self.logstddev) for _ in range(count)]

    def nominal_kwargs(self):
        return {
            "logmean": f'SRC_STATS["auto__{self.table_name}"]["results"][0]["logmean__{self.column_name}"]',
            "logsd": f'SRC_STATS["auto__{self.table_name}"]["results"][0]["logstddev__{self.column_name}"]',
        }

    def actual_kwargs(self):
        return {
            "logmean": self.logmean,
            "logsd": self.logstddev,
        }

    def select_aggregate_clauses(self) -> dict[str, dict[str, str]]:
        clauses = super().select_aggregate_clauses()
        return {
            **clauses,
            f"logmean__{self.column_name}": {
                "clause": f"AVG(CASE WHEN 0<{self.column_name} THEN LN({self.column_name}) ELSE NULL END)",
                "comment": f"Mean of logs of {self.column_name} from table {self.table_name}",
            },
            f"logstddev__{self.column_name}": {
                "clause": f"STDDEV(CASE WHEN 0<{self.column_name} THEN LN({self.column_name}) ELSE NULL END)",
                "comment": f"Standard deviation of logs of {self.column_name} from table {self.table_name}",
            },
        }

    def fit(self, default=None):
        if self.buckets is None:
            return default
        return self.buckets.fit_from_counts(self.expected_buckets)


class ContinuousLogDistributionGeneratorFactory(ContinuousDistributionGeneratorFactory):
    """
    All generators that want an average and standard deviation of log data.
    """

    def _get_generators_from_buckets(
        self,
        engine: Engine,
        table_name: str,
        column_name: str,
        buckets: Buckets,
    ) -> list[Generator]:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT AVG(CASE WHEN 0<{column} THEN LN({column}) ELSE NULL END) AS logmean, STDDEV(CASE WHEN 0<{column} THEN LN({column}) ELSE NULL END) AS logstddev FROM {table}".format(
                        table=table_name,
                        column=column_name,
                    )
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


def zipf_distribution(total, bins):
    basic_dist = list(map(lambda n: 1 / n, range(1, bins + 1)))
    bd_remaining = sum(basic_dist)
    for b in basic_dist:
        # yield b/bd_remaining of the `total` remaining
        if bd_remaining == 0:
            yield 0
        else:
            x = math.floor(0.5 + total * b / bd_remaining)
            bd_remaining -= x * bd_remaining / total
            total -= x
            yield x


class ChoiceGenerator(Generator):
    STORE_COUNTS = False

    def __init__(
        self,
        table_name,
        column_name,
        values,
        counts,
        sample_count=None,
        suppress_count=0,
    ):
        super().__init__()
        self.table_name = table_name
        self.column_name = column_name
        self.values = values
        estimated_counts = self.get_estimated_counts(counts)
        self._fit = fit_from_buckets(counts, estimated_counts)

        extra_results = ""
        extra_expo = ""
        extra_comment = ""
        if self.STORE_COUNTS:
            extra_results = f", COUNT({column_name}) AS count"
            extra_expo = ", count"
            extra_comment = " and their counts"
        if suppress_count == 0:
            if sample_count is None:
                self._query = f"SELECT {column_name} AS value{extra_results} FROM {table_name} WHERE {column_name} IS NOT NULL GROUP BY value ORDER BY COUNT({column_name}) DESC"
                self._comment = f"All the values{extra_comment} that appear in column {column_name} of table {table_name}"
                self._annotation = None
            else:
                self._query = f"SELECT {column_name} AS value{extra_results} FROM (SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL ORDER BY RANDOM() LIMIT {sample_count}) AS _inner GROUP BY value ORDER BY COUNT({column_name}) DESC"
                self._comment = f"The values{extra_comment} that appear in column {column_name} of a random sample of {sample_count} rows of table {table_name}"
                self._annotation = "sampled"
        else:
            if sample_count is None:
                self._query = f"SELECT value{extra_expo} FROM (SELECT {column_name} AS value, COUNT({column_name}) AS count FROM {table_name} WHERE {column_name} IS NOT NULL GROUP BY value ORDER BY count DESC) AS _inner WHERE {suppress_count} < count"
                self._comment = f"All the values{extra_comment} that appear in column {column_name} of table {table_name} more than {suppress_count} times"
                self._annotation = "suppressed"
            else:
                self._query = f"SELECT value{extra_expo} FROM (SELECT value, COUNT(value) AS count FROM (SELECT {column_name} AS value FROM {table_name} WHERE {column_name} IS NOT NULL ORDER BY RANDOM() LIMIT {sample_count}) AS _inner GROUP BY value ORDER BY count DESC) AS _inner WHERE {suppress_count} < count"
                self._comment = f"The values{extra_comment} that appear more than {suppress_count} times in column {column_name}, out of a random sample of {sample_count} rows of table {table_name}"
                self._annotation = "sampled and suppressed"

    @abstractmethod
    def get_estimated_counts(self, counts):
        """
        The counts that we would expect if this distribution was the correct one.
        """

    def nominal_kwargs(self):
        return {
            "a": f'SRC_STATS["auto__{self.table_name}__{self.column_name}"]["results"]',
        }

    def name(self) -> str:
        n = super().name()
        if self._annotation is None:
            return n
        return f"{n} [{self._annotation}]"

    def actual_kwargs(self):
        return {
            "a": self.values,
        }

    def custom_queries(self) -> dict[str, dict[str, str]]:
        qs = super().custom_queries()
        return {
            **qs,
            f"auto__{self.table_name}__{self.column_name}": {
                "query": self._query,
                "comment": self._comment,
            },
        }

    def fit(self, default=None) -> float | None:
        return default if self._fit is None else self._fit


class ZipfChoiceGenerator(ChoiceGenerator):
    def get_estimated_counts(self, counts: list[int]) -> list[int]:
        return list(zipf_distribution(sum(counts), len(counts)))

    def function_name(self) -> str:
        return "dist_gen.zipf_choice"

    def generate_data(self, count: int) -> list[float]:
        return [
            dist_gen.zipf_choice(self.values, len(self.values)) for _ in range(count)
        ]


def uniform_distribution(total, bins: int) -> typing.Generator[int, None, None]:
    p = total // bins
    n = total % bins
    for _ in range(0, n):
        yield p + 1
    for _ in range(n, bins):
        yield p


class UniformChoiceGenerator(ChoiceGenerator):
    def get_estimated_counts(self, counts):
        return list(uniform_distribution(sum(counts), len(counts)))

    def function_name(self):
        return "dist_gen.choice"

    def generate_data(self, count):
        return [dist_gen.choice(self.values) for _ in range(count)]


class WeightedChoiceGenerator(ChoiceGenerator):
    STORE_COUNTS = True

    def get_estimated_counts(self, counts):
        return counts

    def function_name(self):
        return "dist_gen.weighted_choice"

    def generate_data(self, count):
        return [dist_gen.weighted_choice(self.values) for _ in range(count)]


class ChoiceGeneratorFactory(GeneratorFactory):
    """
    All generators that want an average and standard deviation.
    """

    SAMPLE_COUNT = MAXIMUM_CHOICES
    SUPPRESS_COUNT = 5

    def get_generators(self, columns: list[Column], engine: Engine):
        if len(columns) != 1:
            return []
        column = columns[0]
        column_name = column.name
        table_name = column.table.name
        generators = []
        with engine.connect() as connection:
            results = connection.execute(
                text(
                    "SELECT {column} AS v, COUNT({column}) AS f FROM {table} GROUP BY v ORDER BY f DESC LIMIT {limit}".format(
                        table=table_name,
                        column=column_name,
                        limit=MAXIMUM_CHOICES + 1,
                    )
                )
            )
            if results is not None and results.rowcount <= MAXIMUM_CHOICES:
                values = []  # The values found
                counts = []  # The number or each value
                cvs: list[
                    dict[str, Any]
                ] = []  # list of dicts with keys "v" and "count"
                for result in results:
                    c = result.f
                    if c != 0:
                        counts.append(c)
                        v = result.v
                        if type(v) is decimal.Decimal:
                            v = float(v)
                        values.append(v)
                        cvs.append({"value": v, "count": c})
                if counts:
                    generators += [
                        ZipfChoiceGenerator(table_name, column_name, values, counts),
                        UniformChoiceGenerator(table_name, column_name, values, counts),
                        WeightedChoiceGenerator(table_name, column_name, cvs, counts),
                    ]
            results = connection.execute(
                text(
                    "SELECT v, COUNT(v) AS f FROM (SELECT {column} as v FROM {table} ORDER BY RANDOM() LIMIT {sample_count}) AS _inner GROUP BY v ORDER BY f DESC".format(
                        table=table_name,
                        column=column_name,
                        sample_count=self.SAMPLE_COUNT,
                    )
                )
            )
            if results is not None:
                values = []  # All values found
                counts = []  # The number or each value
                cvs: list[
                    dict[str, Any]
                ] = []  # list of dicts with keys "v" and "count"
                values_not_suppressed = (
                    []
                )  # All values found more than SUPPRESS_COUNT times
                counts_not_suppressed = []  # The number for each value not suppressed
                cvs_not_suppressed: list[
                    dict[str, Any]
                ] = []  # list of dicts with keys "v" and "count"
                for result in results:
                    c = result.f
                    if c != 0:
                        counts.append(c)
                        v = result.v
                        if type(v) is decimal.Decimal:
                            v = float(v)
                        values.append(v)
                        cvs.append({"value": v, "count": c})
                    if self.SUPPRESS_COUNT < c:
                        counts_not_suppressed.append(c)
                        v = result.v
                        if type(v) is decimal.Decimal:
                            v = float(v)
                        values_not_suppressed.append(v)
                        cvs_not_suppressed.append({"value": v, "count": c})
                if counts:
                    generators += [
                        ZipfChoiceGenerator(
                            table_name,
                            column_name,
                            values,
                            counts,
                            sample_count=self.SAMPLE_COUNT,
                        ),
                        UniformChoiceGenerator(
                            table_name,
                            column_name,
                            values,
                            counts,
                            sample_count=self.SAMPLE_COUNT,
                        ),
                        WeightedChoiceGenerator(
                            table_name,
                            column_name,
                            cvs,
                            counts,
                            sample_count=self.SAMPLE_COUNT,
                        ),
                    ]
                if counts_not_suppressed:
                    generators += [
                        ZipfChoiceGenerator(
                            table_name,
                            column_name,
                            values_not_suppressed,
                            counts_not_suppressed,
                            sample_count=self.SAMPLE_COUNT,
                            suppress_count=self.SUPPRESS_COUNT,
                        ),
                        UniformChoiceGenerator(
                            table_name,
                            column_name,
                            values_not_suppressed,
                            counts_not_suppressed,
                            sample_count=self.SAMPLE_COUNT,
                            suppress_count=self.SUPPRESS_COUNT,
                        ),
                        WeightedChoiceGenerator(
                            table_name=table_name,
                            column_name=column_name,
                            values=cvs_not_suppressed,
                            counts=counts,
                            sample_count=self.SAMPLE_COUNT,
                            suppress_count=self.SUPPRESS_COUNT,
                        ),
                    ]
        return generators


class ConstantGenerator(Generator):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.repr = repr(value)

    def function_name(self) -> str:
        return "dist_gen.constant"

    def nominal_kwargs(self) -> dict[str, str]:
        return {"value": self.repr}

    def actual_kwargs(self) -> dict[str, Any]:
        return {"value": self.value}

    def generate_data(self, count) -> list[Any]:
        return [self.value for _ in range(count)]


class ConstantGeneratorFactory(GeneratorFactory):
    """
    Just the null generator
    """

    def get_generators(self, columns: list[Column], engine: Engine):
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


class MultivariateNormalGenerator(Generator):
    def __init__(
        self,
        table_name: list[str],
        column_names: list[str],
        query: str,
        covariates: dict[str, float],
        function_name: str,
    ):
        self._table = table_name
        self._columns = column_names
        self._query = query
        self._covariates = covariates
        self._function_name = function_name

    def function_name(self):
        return "dist_gen." + self._function_name

    def nominal_kwargs(self):
        return {
            "cov": f'SRC_STATS["auto__cov__{self._table}"]["results"][0]',
        }

    def custom_queries(self):
        cols = ", ".join(self._columns)
        return {
            f"auto__cov__{self._table}": {
                "comment": f"Means and covariate matrix for the columns {cols}, so that we can produce the relatedness between these in the fake data.",
                "query": self._query,
            }
        }

    def actual_kwargs(self) -> dict[str, Any]:
        """
        The kwargs (summary statistics) this generator is instantiated with.
        """
        return {"cov": self._covariates}

    def generate_data(self, count) -> list[Any]:
        """
        Generate 'count' random data points for this column.
        """
        return [
            getattr(dist_gen, self._function_name)(self._covariates)
            for _ in range(count)
        ]

    def fit(self, default=None) -> float | None:
        return default


class MultivariateNormalGeneratorFactory(GeneratorFactory):
    def function_name(self) -> str:
        return "multivariate_normal"

    def query_predicate(self, column: Column) -> str:
        return column.name + " IS NOT NULL"

    def query_var(self, column: str) -> str:
        return column

    def query(
        self,
        table: str,
        columns: list[Column],
        predicates: list[str] = [],
        group_by_clause: str = "",
        constant_clauses: str = "",
        constants: str = "",
        suppress_count: int = 1,
        sample_count: int | None = None,
    ) -> str:
        """
        Gets a query for the basics for multivariate normal/lognormal parameters.
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
        preds = [self.query_predicate(col) for col in columns] + predicates
        where = " WHERE " + " AND ".join(preds) if preds else ""
        avgs = "".join(
            f", AVG({self.query_var(col.name)}) AS m{i}"
            for i, col in enumerate(columns)
        )
        multiples = "".join(
            f", SUM({self.query_var(colx.name)} * {self.query_var(coly.name)}) AS s{ix}_{iy}"
            for iy, coly in enumerate(columns)
            for ix, colx in enumerate(columns[: iy + 1])
        )
        means = "".join(f", _q.m{i}" for i in range(len(columns)))
        covs = "".join(
            f", (_q.s{ix}_{iy} - _q.count * _q.m{ix} * _q.m{iy})/NULLIF(_q.count - 1, 0) AS c{ix}_{iy}"
            for iy in range(len(columns))
            for ix in range(iy + 1)
        )
        if sample_count is None:
            subquery = table + where
        else:
            subquery = f"(SELECT * FROM {table}{where} ORDER BY RANDOM() LIMIT {sample_count}) AS _sampled"
        # if there are any numeric columns we need at least two rows to make any (co)variances at all
        suppress_clause = f" WHERE {suppress_count} < _q.count" if columns else ""
        return (
            f"SELECT {len(columns)} AS rank{constant_clauses}, _q.count AS count{means}{covs}"
            f" FROM (SELECT COUNT(*) AS count{multiples}{avgs}{constants}"
            f" FROM {subquery}{group_by_clause}) AS _q{suppress_clause}"
        )

    def get_generators(self, columns: list[Column], engine: Engine) -> list[Generator]:
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
            except Exception as e:
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
    def function_name(self) -> str:
        return "multivariate_lognormal"

    def query_predicate(self, column: Column) -> str:
        return f"COALESCE(0 < {column.name}, FALSE)"

    def query_var(self, column: str) -> str:
        return f"LN({column})"


def text_list(items: list[str]) -> str:
    """
    Concatenate the items with commas and one "and".
    """
    if not hasattr(items, "__getitem__"):
        items = list(items)
    if len(items) == 0:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


@dataclass
class RowPartition:
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
    covariates: list[dict[str, float]]

    def comment(self) -> str:
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

    def __init__(
        self,
        query_name: str,
        partitions: dict[int, RowPartition],
        function_name: str = "grouped_multivariate_lognormal",
        name_suffix: str | None = None,
        partition_count_query: str | None = None,
        partition_counts: Sequence[RowMapping] | None = None,
        partition_count_comment: str | None = None,
    ):
        self._query_name = query_name
        self._partitions = partitions
        self._function_name = function_name
        self._partition_count_query = partition_count_query
        self._partition_counts = [dict(pc) for pc in partition_counts]
        self._partition_count_comment = partition_count_comment
        if name_suffix:
            self._name = f"null-partitioned {function_name} [{name_suffix}]"
        else:
            self._name = f"null-partitioned {function_name}"

    def name(self) -> str:
        return self._name

    def function_name(self) -> str:
        return "dist_gen.alternatives"

    def _nominal_kwargs_with_combinations(self, index: int, partition: RowPartition):
        count = f'sum(r["count"] for r in SRC_STATS["auto__cov__{self._query_name}__alt_{index}"]["results"])'
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

    def _count_query_name(self):
        if self._partition_count_query:
            return f"auto__cov__{self._query_name}__counts"
        return None

    def nominal_kwargs(self):
        return {
            "alternative_configs": [
                self._nominal_kwargs_with_combinations(index, self._partitions[index])
                for index in range(len(self._partitions))
            ],
            "counts": f'SRC_STATS["{self._count_query_name()}"]["results"]',
        }

    def custom_queries(self):
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
                "comment": self._partition_count_comment,
                "query": self._partition_count_query,
            },
            **partitions,
        }

    def _actual_kwargs_with_combinations(self, partition: RowPartition):
        count = sum(row["count"] for row in partition.covariates)
        if not partition.included_numeric and not partition.included_choice:
            return {
                "count": count,
                "name": "constant",
                "params": {"value": [None] * len(partition.excluded_columns)},
            }
        if not partition.excluded_columns:
            return {
                "count": count,
                "name": self._function_name,
                "params": {
                    "covs": partition.covariates,
                },
            }
        return {
            "count": count,
            "name": "with_constants_at",
            "params": {
                "constants_at": partition.constant_outputs,
                "subgen": self._function_name,
                "params": {
                    "covs": partition.covariates,
                },
            },
        }

    def actual_kwargs(self) -> dict[str, Any]:
        """
        The kwargs (summary statistics) this generator is instantiated with.
        """
        return {
            "alternative_configs": [
                self._actual_kwargs_with_combinations(self._partitions[index])
                for index in range(len(self._partitions))
            ],
            "counts": self._partition_counts,
        }

    def generate_data(self, count) -> list[Any]:
        """
        Generate 'count' random data points for this column.
        """
        kwargs = self.actual_kwargs()
        return [dist_gen.alternatives(**kwargs) for _ in range(count)]

    def fit(self, default=None) -> float | None:
        return default


def is_numeric(col: Column) -> bool:
    ct = get_column_type(col)
    return (isinstance(ct, Numeric) or isinstance(ct, Integer)) and not col.foreign_keys


T = TypeVar("T")


def powerset(input: list[T]) -> Iterable[Iterable[T]]:
    """Returns a list of all sublists of"""
    return chain.from_iterable(combinations(input, n) for n in range(len(input) + 1))


@dataclass
class NullableColumn:
    """
    A reference to a nullable column whose nullability is part of a partitioning.
    """

    column: Column
    # The bit (power of two) of the number of the partition in the partition sizes list
    bitmask: int


class NullPatternPartition:
    """
    The definition of a partition (in other words, what makes it not another partition)
    """

    def __init__(
        self, columns: Iterable[Column], partition_nonnulls: Iterable[NullableColumn]
    ):
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
    SAMPLE_COUNT = MAXIMUM_CHOICES
    SUPPRESS_COUNT = 5

    def function_name(self) -> str:
        return "grouped_multivariate_normal"

    def query_predicate(self, column: Column) -> str:
        """
        Returns a SQL expression that is true when ``column`` is available for analysis.
        """
        if is_numeric(column):
            # x <> x + 1 ensures that x is not infinity or NaN
            return f"COALESCE({column.name} <> {column.name} + 1, FALSE)"
        return f"{column.name} IS NOT NULL"

    def query_var(self, column: str) -> str:
        return column

    def get_nullable_columns(self, columns: list[Column]) -> list[NullableColumn]:
        """
        Gets a list of nullable columns together with bitmasks.
        """
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
        Returns a SQL expression returning columns ``count`` and ``index``.

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
        return f'SELECT count, "index" FROM (SELECT COUNT(*) AS count, {index_exp} AS "index" FROM {table} GROUP BY "index") AS _q {where}'

    def get_generators(self, columns: list[Column], engine: Engine) -> list[Generator]:
        if len(columns) < 2:
            return []
        nullable_columns = self.get_nullable_columns(columns)
        if not nullable_columns:
            return []
        table = columns[0].table.name
        query_name = f"{table}__{columns[0].name}"
        # Partitions for minimal suppression and no sampling
        row_partitions_maximal: dict[int, RowPartition] = {}
        # Partitions for normal suppression and severe sampling
        row_partitions_ss: dict[int, RowPartition] = {}
        for partition_nonnulls in powerset(nullable_columns):
            partition_def = NullPatternPartition(columns, partition_nonnulls)
            query = self.query(
                table=table,
                columns=partition_def.included_numeric,
                predicates=partition_def.predicates,
                group_by_clause=partition_def.group_by_clause,
                constants=partition_def.constants,
                constant_clauses=partition_def.constant_clauses,
            )
            row_partitions_maximal[partition_def.index] = RowPartition(
                query,
                partition_def.included_numeric,
                partition_def.included_choice,
                partition_def.excluded,
                partition_def.nones,
                {},
            )
            query = self.query(
                table=table,
                columns=partition_def.included_numeric,
                predicates=partition_def.predicates,
                group_by_clause=partition_def.group_by_clause,
                constants=partition_def.constants,
                constant_clauses=partition_def.constant_clauses,
                suppress_count=self.SUPPRESS_COUNT,
                sample_count=self.SAMPLE_COUNT,
            )
            row_partitions_ss[partition_def.index] = RowPartition(
                query,
                partition_def.included_numeric,
                partition_def.included_choice,
                partition_def.excluded,
                partition_def.nones,
                {},
            )
        gens: list[Generator] = []
        try:
            with engine.connect() as connection:
                partition_query_max = self.get_partition_count_query(
                    nullable_columns, table
                )
                partition_count_max_results = (
                    connection.execute(text(partition_query_max)).mappings().fetchall()
                )
                count_comment = f"Number of rows for each combination of the columns { {nc.column.name for nc in nullable_columns} } of the table {table} being null"
                if self._execute_partition_queries(connection, row_partitions_maximal):
                    gens.append(
                        NullPartitionedNormalGenerator(
                            query_name,
                            row_partitions_maximal,
                            self.function_name(),
                            partition_count_query=partition_query_max,
                            partition_counts=partition_count_max_results,
                            partition_count_comment=count_comment,
                        )
                    )
                partition_query_ss = self.get_partition_count_query(
                    nullable_columns,
                    table,
                    where=f"WHERE {self.SUPPRESS_COUNT} < count",
                )
                partition_count_ss_results = (
                    connection.execute(text(partition_query_ss)).mappings().fetchall()
                )
                if self._execute_partition_queries(connection, row_partitions_ss):
                    gens.append(
                        NullPartitionedNormalGenerator(
                            query_name,
                            row_partitions_ss,
                            self.function_name(),
                            name_suffix="sampled and suppressed",
                            partition_count_query=partition_query_ss,
                            partition_counts=partition_count_ss_results,
                            partition_count_comment=count_comment,
                        )
                    )
        except sqlalchemy.exc.DatabaseError as exc:
            logger.debug("SQL query failed with error %s [%s]", exc, exc.statement)
            return []
        return gens

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
            rp.covariates = connection.execute(text(rp.query)).mappings().fetchall()
            if not rp.covariates or rp.covariates[0]["count"] is None:
                rp.covariates = [{"count": 0}]
            else:
                found_nonzero = True
        return found_nonzero


class NullPartitionedLogNormalGeneratorFactory(NullPartitionedNormalGeneratorFactory):
    def function_name(self) -> str:
        return "grouped_multivariate_lognormal"

    def query_predicate(self, column: Column) -> str:
        if is_numeric(column):
            # x <> x + 1 ensures that x is not infinity or NaN
            return f"COALESCE({column.name} <> {column.name} + 1 AND 0 < {column.name}, FALSE)"
        return f"{column.name} IS NOT NULL"

    def query_var(self, column: str) -> str:
        return f"LN({column})"


@lru_cache(1)
def everything_factory() -> GeneratorFactory:
    return MultiGeneratorFactory(
        [
            MimesisStringGeneratorFactory(),
            MimesisIntegerGeneratorFactory(),
            MimesisFloatGeneratorFactory(),
            MimesisDateGeneratorFactory(),
            MimesisDateTimeGeneratorFactory(),
            MimesisTimeGeneratorFactory(),
            ContinuousDistributionGeneratorFactory(),
            ContinuousLogDistributionGeneratorFactory(),
            ChoiceGeneratorFactory(),
            ConstantGeneratorFactory(),
            MultivariateNormalGeneratorFactory(),
            MultivariateLogNormalGeneratorFactory(),
            NullPartitionedNormalGeneratorFactory(),
            NullPartitionedLogNormalGeneratorFactory(),
        ]
    )
