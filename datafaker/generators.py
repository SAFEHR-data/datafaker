"""
Generator factories for making generators for single columns.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
import decimal
from functools import lru_cache
import math
import mimesis
import mimesis.locales
import re
import sqlalchemy
from sqlalchemy import Column, Engine, text
from sqlalchemy.types import Date, DateTime, Integer, Numeric, String, Time
from typing import Callable

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
        """ The name of the generator function to put into df.py. """

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

    def custom_queries(self) -> dict[str, str]:
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
    def actual_kwargs(self) -> dict[str, any]:
        """
        The kwargs (summary statistics) this generator is instantiated with.
        """

    @abstractmethod
    def generate_data(self, count) -> list[any]:
        """
        Generate 'count' random data points for this column.
        """

    def fit(self, default=None) -> float | None:
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
    SRC_STAT_NAME_RE = re.compile(r'SRC_STATS\["([^]]*)"\].*')

    def __init__(self, table_name: str, generator_object: Mapping[str, any], config: Mapping[str, any]):
        """
        Initialise a generator from a config.yaml.
        :param config: The entire configuration.
        :param generator_object: The part of the configuration at tables.*.row_generators
        """
        logger.debug("Creating a PredefinedGenerator %s from table %s", generator_object["name"], table_name)
        self._table_name = table_name
        self._name: str = generator_object["name"]
        self._kwn: dict[str, str] = generator_object.get("kwargs", {})
        self._src_stats_mentioned = set()
        for kwnv in self._kwn.values():
            ss = self.SRC_STAT_NAME_RE.match(kwnv)
            if ss:
                ss_name = ss.group(1)
                self._src_stats_mentioned.add(ss_name)
                logger.debug("Found SRC_STATS reference %s", ss_name)
            else:
                logger.debug("Value %s does not seem to be a SRC_STATS reference", kwnv)
        # Need to deal with this somehow (or remove it from the schema)
        self._argn: list[str] = generator_object.get("args", [])
        self._select_aggregate_clauses = {}
        self._custom_queries = {}
        for sstat in config.get("src-stats", []):
            name: str = sstat["name"]
            dpq = sstat.get("dp-query", None)
            query = sstat.get("query", dpq)  #... should we really be combining query and dp-query?
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
                        for clause in sam.group(1).split(',')
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

    def actual_kwargs(self) -> dict[str, any]:
        # Run the queries from nominal_kwargs
        #...
        logger.error("PredefinedGenerator.actual_kwargs not implemented yet")
        return {}

    def generate_data(self, count) -> list[any]:
        # Call the function if we can. This could be tricky...
        #...
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
    def __init__(self, engine: Engine, table_name: str, column_name: str, mean:float, stddev: float, count: int):
        with engine.connect() as connection:
            raw_buckets = connection.execute(text(
                "SELECT COUNT({column}) AS f, FLOOR(({column} - {x})/{w}) AS b FROM {table} GROUP BY b".format(
                    column=column_name, table=table_name, x=mean - 2 * stddev, w = stddev / 2
                )
            ))
            self.buckets = [0] * 10
            for rb in raw_buckets:
                if rb.b is not None:
                    bucket = min(9, max(0, int(rb.b) + 1))
                    self.buckets[bucket] += rb.f / count
            self.mean = mean
            self.stddev = stddev

    @classmethod
    def make_buckets(_cls, engine: Engine, table_name: str, column_name: str):
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
                text("SELECT AVG({column}) AS mean, STDDEV({column}) AS stddev, COUNT({column}) AS count FROM {table}".format(
                    table=table_name,
                    column=column_name,
                ))
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
            b = min(9, max(0, int((v - x)/w)))
            buckets[b] += 1
        return self.fit_from_counts(buckets)


class MultiGeneratorFactory(GeneratorFactory):
    """ A composite factory. """
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
                raise Exception(f"Mimesis does not have a function {function_name}: {part} not found")
            f = getattr(f, part)
        if not callable(f):
            raise Exception(f"Mimesis object {function_name} is not a callable, so cannot be used as a generator")
        self._name = "generic." + function_name
        self._generator_function = f
    def function_name(self):
        return self._name
    def generate_data(self, count):
        return [
            self._generator_function()
            for _ in range(count)
        ]


class MimesisGenerator(MimesisGeneratorBase):
    def __init__(
        self,
        function_name: str,
        value_fn: Callable[[any], float] | None=None,
        buckets: Buckets | None=None,
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
            samples = [
                value_fn(s)
                for s in samples
            ]
        self._fit = buckets.fit_from_values(samples)
    def function_name(self):
        return self._name
    def nominal_kwargs(self):
        return {}
    def actual_kwargs(self):
        return {}
    def fit(self, default=None):
        return default if self._fit is None else self._fit


class MimesisDateTimeGenerator(MimesisGeneratorBase):
    def __init__(self, column: Column, function_name: str, min_year: str, max_year: str, start: int, end: int):
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
                text(f"SELECT {min_year} AS start, {max_year} AS end FROM {column.table.name}")
            ).first()
            if result is None or result.start is None or result.end is None:
                return []
        return [MimesisDateTimeGenerator(
            column,
            function_name,
            min_year,
            max_year,
            int(result.start),
            int(result.end),
        )]
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
    def get_generators(self, columns: list[Column], engine: Engine):
        if len(columns) != 1:
            return []
        column = columns[0]
        if not isinstance(get_column_type(column), String):
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
        return list(map(lambda gen: MimesisGenerator(gen, fitness_fn, buckets), [
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
        ]))


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
        return list(map(MimesisGenerator, [
            "person.height",
        ]))


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
        return MimesisDateTimeGenerator.make_singleton(column, engine, "datetime.datetime")


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
    sum_diff_squared = sum(map(lambda t, a: (t - a)*(t - a), xs, ys))
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
    expected_buckets = [0.0227, 0.0441, 0.0918, 0.1499, 0.1915, 0.1915, 0.1499, 0.0918, 0.0441, 0.0227]
    def function_name(self):
        return "dist_gen.normal"
    def generate_data(self, count):
        return [
            dist_gen.normal(self.buckets.mean, self.buckets.stddev)
            for _ in range(count)
        ]


class UniformGenerator(ContinuousDistributionGenerator):
    expected_buckets = [0, 0.06698, 0.14434, 0.14434, 0.14434, 0.14434, 0.14434, 0.14434, 0.06698, 0]
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
        return self._get_generators_from_buckets(engine, table_name, column_name, buckets)


class LogNormalGenerator(Generator):
    #TODO: figure out the real buckets here (this was from a random sample in R)
    expected_buckets = [0, 0, 0, 0.28627, 0.40607, 0.14937, 0.06735, 0.03492, 0.01918, 0.03684]
    def __init__(self, table_name: str, column_name: str, buckets: Buckets, logmean: float, logstddev: float):
        super().__init__()
        self.table_name = table_name
        self.column_name = column_name
        self.buckets = buckets
        self.logmean = logmean
        self.logstddev = logstddev
    def function_name(self):
        return "dist_gen.lognormal"
    def generate_data(self, count):
        return [
            dist_gen.lognormal(self.logmean, self.logstddev)
            for _ in range(count)
        ]
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
                text("SELECT AVG(CASE WHEN 0<{column} THEN LN({column}) ELSE NULL END) AS logmean, STDDEV(CASE WHEN 0<{column} THEN LN({column}) ELSE NULL END) AS logstddev FROM {table}".format(
                    table=table_name,
                    column=column_name,
                ))
            ).first()
            if result is None or result.logstddev is None:
                return None
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
    basic_dist = list(map(lambda n: 1/n, range(1, bins + 1)))
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
        sample_count = None,
        suppress_count = 0,
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
    def get_estimated_counts(counts):
        """
        The counts that we would expect if this distribution was the correct one.
        """
    def nominal_kwargs(self):
        return {
            "a": f'SRC_STATS["auto__{self.table_name}__{self.column_name}"]["results"]',
        }
    def name(self):
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
            }
        }
    def fit(self, default=None):
        return default if self._fit is None else self._fit

class ZipfChoiceGenerator(ChoiceGenerator):
    def get_estimated_counts(self, counts):
        return list(zipf_distribution(sum(counts), len(counts)))
    def function_name(self):
        return "dist_gen.zipf_choice"
    def generate_data(self, count):
        return [
            dist_gen.zipf_choice(self.values, len(self.values))
            for _ in range(count)
        ]


def uniform_distribution(total, bins):
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
        return [
            dist_gen.choice(self.values)
            for _ in range(count)
        ]


class WeightedChoiceGenerator(ChoiceGenerator):
    STORE_COUNTS = True
    def get_estimated_counts(self, counts):
        return counts
    def function_name(self):
        return "dist_gen.weighted_choice"
    def generate_data(self, count):
        return [
            dist_gen.weighted_choice(self.values)
            for _ in range(count)
        ]


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
                text("SELECT {column} AS v, COUNT({column}) AS f FROM {table} GROUP BY v ORDER BY f DESC LIMIT {limit}".format(
                    table=table_name,
                    column=column_name,
                    limit=MAXIMUM_CHOICES+1,
                ))
            )
            if results is not None and results.rowcount <= MAXIMUM_CHOICES:
                values = []  # The values found
                counts = []  # The number or each value
                cvs: list[dict[str, any]] = []  # list of dicts with keys "v" and "count"
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
                text("SELECT v, COUNT(v) AS f FROM (SELECT {column} as v FROM {table} ORDER BY RANDOM() LIMIT {sample_count}) AS _inner GROUP BY v ORDER BY f DESC".format(
                    table=table_name,
                    column=column_name,
                    sample_count=self.SAMPLE_COUNT,
                ))
            )
            if results is not None:
                values = []  # All values found
                counts = []  # The number or each value
                cvs: list[dict[str, any]] = []  # list of dicts with keys "v" and "count"
                values_not_suppressed = []  # All values found more than SUPPRESS_COUNT times
                counts_not_suppressed = []  # The number for each value not suppressed
                cvs_not_suppressed: list[dict[str, any]] = []  # list of dicts with keys "v" and "count"
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
                        ZipfChoiceGenerator(table_name, column_name, values, counts, sample_count=self.SAMPLE_COUNT),
                        UniformChoiceGenerator(table_name, column_name, values, counts, sample_count=self.SAMPLE_COUNT),
                        WeightedChoiceGenerator(table_name, column_name, cvs, counts, sample_count=self.SAMPLE_COUNT),
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
    def actual_kwargs(self) -> dict[str, any]:
        return {"value": self.value}
    def generate_data(self, count) -> list[any]:
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

    def actual_kwargs(self) -> dict[str, any]:
        """
        The kwargs (summary statistics) this generator is instantiated with.
        """
        return { "cov": self._covariates }

    def generate_data(self, count) -> list[any]:
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

    def query_predicate(self, column: str) -> str:
        return column + " IS NOT NULL"

    def query_var(self, column: str) -> str:
        return column

    def query(self, table: str, columns: str) -> str:
        preds = " AND ".join(
            self.query_predicate(col)
            for col in columns
        )
        avgs = ", ".join(
            f"AVG({self.query_var(col)}) AS m{i}"
            for i, col in enumerate(columns)
        )
        multiples = ", ".join(
            f"SUM({self.query_var(colx)} * {self.query_var(coly)}) AS s{ix}_{iy}"
            for iy, coly in enumerate(columns)
            for ix, colx in enumerate(columns[:iy+1])
        )
        means = ", ".join(
            f"q.m{i}" for i in range(len(columns))
        )
        covs = ", ".join(
            f"(q.s{ix}_{iy} - q.count * q.m{ix} * q.m{iy})/NULLIF(q.count - 1, 0) AS c{ix}_{iy}"
            for iy in range(len(columns))
            for ix in range(iy+1)
        )
        return (
            f"SELECT {means}, {covs}, {len(columns)} AS rank"
            f" FROM (SELECT COUNT(*) AS count, {multiples}, {avgs}"
            f" FROM {table} WHERE {preds}) AS q"
        )

    def get_generators(self, columns: list[Column], engine: Engine):
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
        query = self.query(table, column_names)
        with engine.connect() as connection:
            covariates = connection.execute(text(
                query
            )).mappings().first()
            if not covariates or covariates["c0_0"] is None:
                return []
            return [MultivariateNormalGenerator(
                table,
                column_names,
                query,
                covariates,
                self.function_name(),
            )]


class MultivariateLogNormalGeneratorFactory(MultivariateNormalGeneratorFactory):
    def function_name(self) -> str:
        return "multivariate_lognormal"

    def query_predicate(self, column: str) -> str:
        return f"{column} IS NOT NULL AND 0 < {column}"

    def query_var(self, column: str) -> str:
        return f"LN({column})"


@lru_cache(1)
def everything_factory():
    return MultiGeneratorFactory([
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
    ])
