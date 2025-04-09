"""
Generator factories for making generators for single columns.
"""

from abc import ABC, abstractmethod
import decimal
import math
import mimesis
import mimesis.locales
from sqlalchemy import Column, Engine, text
from sqlalchemy.types import Integer, Numeric, String
from typing import Callable

from sqlsynthgen.base import DistributionGenerator

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
        """ The name of the generator function to put into ssg.py. """

    @abstractmethod
    def nominal_kwargs(self) -> dict[str, str]:
        """
        The kwargs the generator wants to be called with.
        The values will tend to be references to something in the src-stats.yaml
        file.
        For example {"avg_age": 'SRC_STATS["auto__patient"]["age_mean"]'} will
        provide the value stored in src-stats.yaml as 
        SRC_STATS["auto__patient"]["age_mean"] as the "avg_age" argument
        to the generator function.
        """

    def select_aggregate_clauses(self) -> dict[str, str]:
        """
        SQL clauses to add to a SELECT ... FROM {table} query.

        Will add to SRC_STATS["auto__{table}"]
        For example {"count": "COUNT(*)", "avg_thiscolumn": "AVG(thiscolumn)"}
        will make the clause become:
        "SELECT COUNT(*) AS count, AVG(thiscolumn) AS avg_thiscolumn FROM thistable"
        and this will populate SRC_STATS["auto__thistable"]["count"] and
        SRC_STATS["auto__thistable"]["avg_thiscolumn"] in the src-stats.yaml file.
        """
        return {}

    def custom_queries(self) -> dict[str, str]:
        """
        SQL queries to add to SRC_STATS.

        Should be used for queries that do not follow the SELECT ... FROM table format,
        because these should use select_aggregate_clauses.

        For example {"myquery", "SELECT one, too AS two FROM mytable WHERE too > 1"}
        will populate SRC_STATS["myquery"]["one"] and SRC_STATS["myquery"]["two"]
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


class GeneratorFactory(ABC):
    """
    A factory for making generators appropriate for a database column.
   """
    @abstractmethod
    def get_generators(self, column: Column, engine: Engine) -> list[Generator]:
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
        """
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT AVG({column}) AS mean, STDDEV({column}) AS stddev, COUNT({column}) AS count FROM {table}".format(
                    table=table_name,
                    column=column_name,
                ))
            ).first()
            if result is None:
                return None
        return Buckets(engine, table_name, column_name, result.mean, result.stddev, result.count)

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
        x=self.mean - 2 * self.stddev
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

    def get_generators(self, column: Column, engine: Engine) -> list[Generator]:
        return [
            generator
            for factory in self.factories
            for generator in factory.get_generators(column, engine)
        ]


class MimesisGenerator(Generator):
    def __init__(
        self,
        function_name: str,
        value_fn: Callable[[any], float] | None=None,
        buckets: Buckets | None=None,
    ):
        """
        Generator from Mimesis.

        :param: function_name is relative to 'generic', for example 'person.name'.
        :param: value_fn Function to convert generator output to floats, if needed. The values
        thus produced are compared against the buckets to estimate the fit.
        :param: buckets The distribution of string lengths in the real data. If this is None
        then the fit method will return None.
        """
        super().__init__()
        f = generic
        for part in function_name.split("."):
            if not hasattr(f, part):
                raise Exception(f"Mimesis does not have a function {function_name}: {part} not found")
            f = getattr(f, part)
        if not callable(f):
            raise Exception(f"Mimesis object {function_name} is not a callable, so cannot be used as a generator")
        self.name = "generic." + function_name
        self.generator_function = f
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
        return self.name
    def nominal_kwargs(self):
        return {}
    def actual_kwargs(self):
        return {}
    def generate_data(self, count):
        return [
            self.generator_function()
            for _ in range(count)
        ]
    def fit(self, default=None):
        return default if self._fit is None else self._fit


class MimesisStringGeneratorFactory(GeneratorFactory):
    """
    All Mimesis generators that return strings.
    """
    def get_generators(self, column: Column, engine: Engine):
        if not isinstance(column.type.as_generic(), String):
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
    def get_generators(self, column: Column, _engine: Engine):
        if not isinstance(column.type.as_generic(), Numeric):
            return []
        return list(map(MimesisGenerator, [
            "person.height",
        ]))

class MimesisIntegerGeneratorFactory(GeneratorFactory):
    """
    All Mimesis generators that return integers.
    """
    def get_generators(self, column: Column, _engine: Engine):
        ct = column.type.as_generic()
        if not isinstance(ct, Numeric) and not isinstance(ct, Integer):
            return []
        return list(map(MimesisGenerator, [
            "person.weight",
            "person.age",
        ]))


def fit_from_buckets(xs: list[float], ys: list[float]):
    sum_diff_squared = sum(map(lambda t, a: (t - a)*(t - a), xs, ys))
    return sum_diff_squared / len(ys)


class ContinuousDistributionGenerator(Generator):
    def __init__(self, table_name: str, column_name: str, buckets: Buckets):
        super().__init__()
        self.table_name = table_name
        self.column_name = column_name
        self.buckets = buckets
    def nominal_kwargs(self):
        return {
            "mean": f'SRC_STATS["auto__{self.table_name}"]["mean__{self.column_name}"]',
            "sd": f'SRC_STATS["auto__{self.table_name}"]["stddev__{self.column_name}"]',
        }
    def actual_kwargs(self):
        if self.buckets is None:
            return {}
        return {
            "mean": self.buckets.mean,
            "sd": self.buckets.stddev,
        }
    def select_aggregate_clauses(self):
        clauses = super().select_aggregate_clauses()
        return {
            **clauses,
            f"mean__{self.column_name}": f"AVG({self.column_name})",
            f"stddev__{self.column_name}": f"STDDEV({self.column_name})",
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
    def get_generators(self, column: Column, engine: Engine):
        ct = column.type.as_generic()
        if not isinstance(ct, Numeric) and not isinstance(ct, Integer):
            return []
        column_name = column.name
        table_name = column.table.name
        buckets = Buckets.make_buckets(engine, table_name, column_name)
        return [
            GaussianGenerator(table_name, column_name, buckets),
            UniformGenerator(table_name, column_name, buckets),
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
    def __init__(self, table_name, column_name, values, counts):
        super().__init__()
        self.table_name = table_name
        self.column_name = column_name
        self.values = values
        estimated_counts = self.get_estimated_counts(counts)
        self._fit = fit_from_buckets(counts, estimated_counts)
    def nominal_kwargs(self):
        return {
            "a": f'SRC_STATS["auto__{self.table_name}__{self.column_name}"]["value"]',
        }
    def actual_kwargs(self):
        return {
            "a": self.values,
        }
    def custom_queries(self):
        qs = super().custom_queries()
        t = self.table_name
        c = self.column_name
        return {
            **qs,
            f"auto__{t}__{c}": f"SELECT {c} AS value FROM {t} GROUP BY value ORDER BY COUNT({c}) DESC",
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
    for i in range(0, n):
        yield p + 1
    for i in range(n, bins):
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


class ChoiceGeneratorFactory(GeneratorFactory):
    """
    All generators that want an average and standard deviation.
    """
    def get_generators(self, column, engine: Engine):
        ct = column.type.as_generic()
        column_name = column.name
        table_name = column.table.name
        with engine.connect() as connection:
            results = connection.execute(
                text("SELECT {column} AS v, COUNT({column}) AS f FROM {table} GROUP BY v ORDER BY f DESC LIMIT {limit}".format(
                    table=table_name,
                    column=column_name,
                    limit=MAXIMUM_CHOICES+1,
                ))
            )
            if results is None or MAXIMUM_CHOICES < results.rowcount:
                return []
            values = []  # The values found
            counts = []  # The number or each value
            total = 0  # total number of non-NULL results
            for result in results:
                c = result.f
                if c != 0:
                    total += c
                    counts.append(c)
                    v = result.v
                    if type(v) is decimal.Decimal:
                        v = float(v)
                    values.append(v)
        if not counts:
            return []
        return [
            ZipfChoiceGenerator(table_name, column_name, values, counts),
            UniformChoiceGenerator(table_name, column_name, values, counts),
        ]


everything_factory = MultiGeneratorFactory([
    MimesisStringGeneratorFactory(),
    MimesisIntegerGeneratorFactory(),
    MimesisFloatGeneratorFactory(),
    ContinuousDistributionGeneratorFactory(),
    ChoiceGeneratorFactory(),
])
