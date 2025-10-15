"""Generators using Mimesis."""

from typing import Any, Callable, Sequence, Union

import mimesis
import mimesis.locales
from sqlalchemy import Column, Engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import Date, DateTime, Integer, Numeric, String, Time

from datafaker.generators.base import (
    Buckets,
    DistributionGenerator,
    Generator,
    GeneratorError,
    GeneratorFactory,
    get_column_type,
)

NumericType = Union[int, float]

# How many distinct values can we have before we consider a
# choice distribution to be infeasible?
MAXIMUM_CHOICES = 500

dist_gen = DistributionGenerator()
generic = mimesis.Generic(locale=mimesis.locales.Locale.EN_GB)


class MimesisGeneratorBase(Generator):
    """Base class for a generator using Mimesis."""

    def __init__(
        self,
        function_name: str,
    ):
        """
        Initialise a generator that uses Mimesis.

        :param function_name: is relative to 'generic', for example 'person.name'.
        """
        super().__init__()
        f = generic
        for part in function_name.split("."):
            if not hasattr(f, part):
                raise GeneratorError(
                    f"Mimesis does not have a function {function_name}: {part} not found"
                )
            f = getattr(f, part)
        if not callable(f):
            raise GeneratorError(
                f"Mimesis object {function_name} is not a callable,"
                " so cannot be used as a generator"
            )
        self._name = "generic." + function_name
        self._generator_function = f

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return self._name

    def generate_data(self, count: int) -> list[Any]:
        """Generate ``count`` random data points for this column."""
        return [self._generator_function() for _ in range(count)]


class MimesisGenerator(MimesisGeneratorBase):
    """A generator using Mimesis."""

    def __init__(
        self,
        function_name: str,
        value_fn: Callable[[Any], float] | None = None,
        buckets: Buckets | None = None,
    ):
        """
        Initialise a generator using Mimesis.

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

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return self._name

    def nominal_kwargs(self) -> dict[str, Any]:
        """Get the arguments to be entered into ``config.yaml``."""
        return {}

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        return {}

    def fit(self, default: float = -1) -> float:
        """Get this generator's fit against the real data."""
        return default if self._fit is None else self._fit


class MimesisGeneratorTruncated(MimesisGenerator):
    """A string generator using Mimesis that must fit within a certain number of characters."""

    def __init__(
        self,
        function_name: str,
        length: int,
        value_fn: Callable[[Any], float] | None = None,
        buckets: Buckets | None = None,
    ):
        """Initialise a MimesisGeneratorTruncated."""
        self._length = length
        super().__init__(function_name, value_fn, buckets)

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "dist_gen.truncated_string"

    def name(self) -> str:
        """Get the name of the generator."""
        return f"{self._name} [truncated to {self._length}]"

    def nominal_kwargs(self) -> dict[str, Any]:
        """Get the arguments to be entered into ``config.yaml``."""
        return {
            "subgen_fn": self._name,
            "params": {},
            "length": self._length,
        }

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        return {
            "subgen_fn": self._name,
            "params": {},
            "length": self._length,
        }

    def generate_data(self, count: int) -> list[Any]:
        """Generate ``count`` random data points for this column."""
        return [self._generator_function()[: self._length] for _ in range(count)]


class MimesisDateTimeGenerator(MimesisGeneratorBase):
    """DateTime generator using Mimesis."""

    # pylint: disable=too-many-arguments too-many-positional-arguments
    def __init__(
        self,
        column: Column,
        function_name: str,
        min_year: str,
        max_year: str,
        start: int,
        end: int,
    ) -> None:
        """
        Initialise a MimesisDateTimeGenerator.

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
    def make_singleton(
        cls, column: Column, engine: Engine, function_name: str
    ) -> Sequence[Generator]:
        """Make the appropriate generation configuration for this column."""
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

    def nominal_kwargs(self) -> dict[str, Any]:
        """Get the arguments to be entered into ``config.yaml``."""
        return {
            "start": (
                f'SRC_STATS["auto__{self._column.table.name}"]["results"]'
                f'[0]["{self._column.name}__start"]'
            ),
            "end": (
                f'SRC_STATS["auto__{self._column.table.name}"]["results"]'
                f'[0]["{self._column.name}__end"]'
            ),
        }

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        return {
            "start": self._start,
            "end": self._end,
        }

    def select_aggregate_clauses(self) -> dict[str, dict[str, str]]:
        """Get the query fragments the generators need to call."""
        return {
            f"{self._column.name}__start": {
                "clause": self._min_year,
                "comment": (
                    f"Earliest year found for column {self._column.name}"
                    f" in table {self._column.table.name}"
                ),
            },
            f"{self._column.name}__end": {
                "clause": self._max_year,
                "comment": (
                    f"Latest year found for column {self._column.name}"
                    f" in table {self._column.table.name}"
                ),
            },
        }

    def generate_data(self, count: int) -> list[Any]:
        """Generate ``count`` random data points for this column."""
        return [
            self._generator_function(start=self._start, end=self._end)
            for _ in range(count)
        ]


class MimesisStringGeneratorFactory(GeneratorFactory):
    """All Mimesis generators that return strings."""

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

    def get_generators(
        self, columns: list[Column], engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators appropriate to these columns."""
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
        except SQLAlchemyError:
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
    """All Mimesis generators that return floating point numbers."""

    def get_generators(
        self, columns: list[Column], _engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators appropriate to these columns."""
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
    """All Mimesis generators that return dates."""

    def get_generators(
        self, columns: list[Column], engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators appropriate to these columns."""
        if len(columns) != 1:
            return []
        column = columns[0]
        ct = get_column_type(column)
        if not isinstance(ct, Date):
            return []
        return MimesisDateTimeGenerator.make_singleton(column, engine, "datetime.date")


class MimesisDateTimeGeneratorFactory(GeneratorFactory):
    """All Mimesis generators that return datetimes."""

    def get_generators(
        self, columns: list[Column], engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators appropriate to these columns."""
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
    """All Mimesis generators that return times."""

    def get_generators(
        self, columns: list[Column], _engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators appropriate to these columns."""
        if len(columns) != 1:
            return []
        column = columns[0]
        ct = get_column_type(column)
        if not isinstance(ct, Time):
            return []
        return [MimesisGenerator("datetime.time")]


class MimesisIntegerGeneratorFactory(GeneratorFactory):
    """All Mimesis generators that return integers."""

    def get_generators(
        self, columns: list[Column], _engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators appropriate to these columns."""
        if len(columns) != 1:
            return []
        column = columns[0]
        ct = get_column_type(column)
        if not isinstance(ct, Numeric) and not isinstance(ct, Integer):
            return []
        return [MimesisGenerator("person.weight")]
