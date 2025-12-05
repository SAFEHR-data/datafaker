"""Generator factories for making generators for choices of values."""

import decimal
import math
import typing
from abc import abstractmethod
from typing import Any, Sequence, Union

from sqlalchemy import Column, CursorResult, Engine, text

from datafaker.generators.base import (
    Generator,
    GeneratorFactory,
    dist_gen,
    fit_from_buckets,
)

NumericType = Union[int, float]

# How many distinct values can we have before we consider a
# choice distribution to be infeasible?
MAXIMUM_CHOICES = 500


def zipf_distribution(total: int, bins: int) -> typing.Generator[int, None, None]:
    """
    Get a zipf distribution for a certain number of items.

    :param total: The total number of items to be distributed.
    :param bins: The total number of bins to distribute the items into.
    :return: A generator of the number of items in each bin, from the
    largest to the smallest.
    """
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
    """Base generator for all generators producing choices of items."""

    STORE_COUNTS = False

    # pylint: disable=too-many-arguments too-many-positional-arguments
    def __init__(
        self,
        table_name: str,
        column_name: str,
        values: list[Any],
        counts: list[int],
        sample_count: int | None = None,
        suppress_count: int = 0,
    ) -> None:
        """Initialise a ChoiceGenerator."""
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
                self._query = (
                    f"SELECT {column_name} AS value{extra_results} FROM {table_name}"
                    f" WHERE {column_name} IS NOT NULL GROUP BY value"
                    f" ORDER BY COUNT({column_name}) DESC"
                )
                self._comment = (
                    f"All the values{extra_comment} that appear in column {column_name}"
                    f" of table {table_name}"
                )
                self._annotation = None
            else:
                self._query = (
                    f"SELECT {column_name} AS value{extra_results} FROM"
                    f" (SELECT {column_name} FROM {table_name}"
                    f" WHERE {column_name} IS NOT NULL"
                    f" ORDER BY RANDOM() LIMIT {sample_count})"
                    f" AS _inner GROUP BY value ORDER BY COUNT({column_name}) DESC"
                )
                self._comment = (
                    f"The values{extra_comment} that appear in column {column_name}"
                    f" of a random sample of {sample_count} rows of table {table_name}"
                )
                self._annotation = "sampled"
        else:
            if sample_count is None:
                self._query = (
                    f"SELECT value{extra_expo} FROM"
                    f" (SELECT {column_name} AS value, COUNT({column_name}) AS count"
                    f" FROM {table_name} WHERE {column_name} IS NOT NULL"
                    f" GROUP BY value ORDER BY count DESC) AS _inner"
                    f" WHERE {suppress_count} < count"
                )
                self._comment = (
                    f"All the values{extra_comment} that appear in column {column_name}"
                    f" of table {table_name} more than {suppress_count} times"
                )
                self._annotation = "suppressed"
            else:
                self._query = (
                    f"SELECT value{extra_expo} FROM (SELECT value, COUNT(value) AS count FROM"
                    f" (SELECT {column_name} AS value FROM {table_name}"
                    f" WHERE {column_name} IS NOT NULL ORDER BY RANDOM() LIMIT {sample_count})"
                    f" AS _inner GROUP BY value ORDER BY count DESC)"
                    f" AS _inner WHERE {suppress_count} < count"
                )
                self._comment = (
                    f"The values{extra_comment} that appear more than {suppress_count} times"
                    f" in column {column_name}, out of a random sample of {sample_count} rows"
                    f" of table {table_name}"
                )
                self._annotation = "sampled and suppressed"

    @abstractmethod
    def get_estimated_counts(self, counts: list[int]) -> list[int]:
        """Get the counts that we would expect if this distribution was the correct one."""

    def nominal_kwargs(self) -> dict[str, Any]:
        """Get the arguments to be entered into ``config.yaml``."""
        return {
            "a": f'SRC_STATS["auto__{self.table_name}__{self.column_name}"]["results"]',
        }

    def name(self) -> str:
        """Get the name of the generator."""
        n = super().name()
        if self._annotation is None:
            return n
        return f"{n} [{self._annotation}]"

    def actual_kwargs(self) -> dict[str, Any]:
        """Get the kwargs (summary statistics) this generator was instantiated with."""
        return {
            "a": self.values,
        }

    def custom_queries(self) -> dict[str, dict[str, Any]]:
        """Get the queries the generators need to call."""
        qs = super().custom_queries()
        return {
            **qs,
            f"auto__{self.table_name}__{self.column_name}": {
                "query": self._query,
                "comments": [self._comment],
            },
        }

    def fit(self, default: float = -1) -> float:
        """Get this generator's fit against the real data."""
        return default if self._fit is None else self._fit


class ZipfChoiceGenerator(ChoiceGenerator):
    """Generator producing items in a Zipf distribution."""

    def get_estimated_counts(self, counts: list[int]) -> list[int]:
        """Get the counts that we would expect if this distribution was the correct one."""
        return list(zipf_distribution(sum(counts), len(counts)))

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "dist_gen.zipf_choice"

    def generate_data(self, count: int) -> list[float]:
        """Generate ``count`` random data points for this column."""
        return [
            dist_gen.zipf_choice_direct(self.values, len(self.values))
            for _ in range(count)
        ]


def uniform_distribution(total: int, bins: int) -> typing.Generator[int, None, None]:
    """
    Construct a distribution putting ``total`` items uniformly into ``bins`` bins.

    If they don't fit exactly evenly, the earlier bins will have one more
    item than the later bins so the total is as required.
    """
    p = total // bins
    n = total % bins
    for _ in range(0, n):
        yield p + 1
    for _ in range(n, bins):
        yield p


class UniformChoiceGenerator(ChoiceGenerator):
    """A generator producing values, each roughly as frequently as each other."""

    def get_estimated_counts(self, counts: list[int]) -> list[int]:
        """Get the counts that we would expect if this distribution was the correct one."""
        return list(uniform_distribution(sum(counts), len(counts)))

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "dist_gen.choice"

    def generate_data(self, count: int) -> list[Any]:
        """Generate ``count`` random data points for this column."""
        return [dist_gen.choice_direct(self.values) for _ in range(count)]


class WeightedChoiceGenerator(ChoiceGenerator):
    """Choice generator that matches the source data's frequency."""

    STORE_COUNTS = True

    def get_estimated_counts(self, counts: list[int]) -> list[int]:
        """Get the counts that we would expect if this distribution was the correct one."""
        return counts

    def function_name(self) -> str:
        """Get the name of the generator function to call."""
        return "dist_gen.weighted_choice"

    def generate_data(self, count: int) -> list[Any]:
        """Generate ``count`` random data points for this column."""
        return [dist_gen.weighted_choice(self.values) for _ in range(count)]


class ValueGatherer:
    """
    Gathers values from a query of values and counts.

    The query must return columns ``v`` for a value and ``f`` for the
    count of how many of those values there are.
    These values will be gathered into a number of properties:
    ``values``: the list of ``v`` values, ``counts``: the list of ``f`` counts
    in the same order as ``v``, ``cvs``: list of dicts with keys ``value`` and
    ``count`` giving these values and counts. ``counts_not_suppressed``,
    ``values_not_suppressed`` and ``cvs_not_suppressed`` are the
    equivalents with the counts less than or equal to ``suppress_count``
    removed.

    :param suppress_count: value with a count of this or fewer will be excluded
    from the suppressed values.
    """

    def __init__(self, results: CursorResult, suppress_count: int = 0) -> None:
        """Initialise a ValueGatherer."""
        values = []  # All values found
        counts = []  # The number or each value
        cvs: list[dict[str, Any]] = []  # list of dicts with keys "v" and "count"
        values_not_suppressed = []  # All values found more than SUPPRESS_COUNT times
        counts_not_suppressed = []  # The number for each value not suppressed
        cvs_not_suppressed: list[
            dict[str, Any]
        ] = []  # list of dicts with keys "v" and "count"
        for result in results:
            c = result.f
            if c != 0:
                counts.append(c)
                v = result.v
                if isinstance(v, decimal.Decimal):
                    v = float(v)
                values.append(v)
                cvs.append({"value": v, "count": c})
            if suppress_count < c:
                counts_not_suppressed.append(c)
                v = result.v
                if isinstance(v, decimal.Decimal):
                    v = float(v)
                values_not_suppressed.append(v)
                cvs_not_suppressed.append({"value": v, "count": c})
        self.values = values
        self.counts = counts
        self.cvs = cvs
        self.values_not_suppressed = values_not_suppressed
        self.counts_not_suppressed = counts_not_suppressed
        self.cvs_not_suppressed = cvs_not_suppressed


class ChoiceGeneratorFactory(GeneratorFactory):
    """All generators that want an average and standard deviation."""

    SAMPLE_COUNT = MAXIMUM_CHOICES
    SUPPRESS_COUNT = 7

    def get_generators(
        self, columns: list[Column], engine: Engine
    ) -> Sequence[Generator]:
        """Get the generators appropriate to these columns."""
        if len(columns) != 1:
            return []
        column = columns[0]
        column_name = column.name
        table_name = column.table.name
        generators = []
        with engine.connect() as connection:
            results = connection.execute(
                text(
                    f'SELECT "{column_name}" AS v, COUNT("{column_name}")'
                    f' AS f FROM "{table_name}" GROUP BY v'
                    f" ORDER BY f DESC LIMIT {MAXIMUM_CHOICES + 1}"
                )
            )
            if results is not None and results.rowcount <= MAXIMUM_CHOICES:
                vg = ValueGatherer(results, self.SUPPRESS_COUNT)
                if vg.counts:
                    generators += [
                        ZipfChoiceGenerator(
                            table_name, column_name, vg.values, vg.counts
                        ),
                        UniformChoiceGenerator(
                            table_name, column_name, vg.values, vg.counts
                        ),
                        WeightedChoiceGenerator(
                            table_name, column_name, vg.cvs, vg.counts
                        ),
                    ]
                if vg.counts_not_suppressed:
                    generators += [
                        ZipfChoiceGenerator(
                            table_name,
                            column_name,
                            vg.values_not_suppressed,
                            vg.counts_not_suppressed,
                            suppress_count=self.SUPPRESS_COUNT,
                        ),
                        UniformChoiceGenerator(
                            table_name,
                            column_name,
                            vg.values_not_suppressed,
                            vg.counts_not_suppressed,
                            suppress_count=self.SUPPRESS_COUNT,
                        ),
                        WeightedChoiceGenerator(
                            table_name=table_name,
                            column_name=column_name,
                            values=vg.cvs_not_suppressed,
                            counts=vg.counts_not_suppressed,
                            suppress_count=self.SUPPRESS_COUNT,
                        ),
                    ]
            sampled_results = connection.execute(
                text(
                    f"SELECT v, COUNT(v) AS f FROM"
                    f' (SELECT "{column_name}" as v FROM "{table_name}"'
                    f" ORDER BY RANDOM() LIMIT {self.SAMPLE_COUNT})"
                    f" AS _inner GROUP BY v ORDER BY f DESC"
                )
            )
            if sampled_results is not None:
                vg = ValueGatherer(sampled_results, self.SUPPRESS_COUNT)
                if vg.counts:
                    generators += [
                        ZipfChoiceGenerator(
                            table_name,
                            column_name,
                            vg.values,
                            vg.counts,
                            sample_count=self.SAMPLE_COUNT,
                        ),
                        UniformChoiceGenerator(
                            table_name,
                            column_name,
                            vg.values,
                            vg.counts,
                            sample_count=self.SAMPLE_COUNT,
                        ),
                        WeightedChoiceGenerator(
                            table_name,
                            column_name,
                            vg.cvs,
                            vg.counts,
                            sample_count=self.SAMPLE_COUNT,
                        ),
                    ]
                if vg.counts_not_suppressed:
                    generators += [
                        ZipfChoiceGenerator(
                            table_name,
                            column_name,
                            vg.values_not_suppressed,
                            vg.counts_not_suppressed,
                            sample_count=self.SAMPLE_COUNT,
                            suppress_count=self.SUPPRESS_COUNT,
                        ),
                        UniformChoiceGenerator(
                            table_name,
                            column_name,
                            vg.values_not_suppressed,
                            vg.counts_not_suppressed,
                            sample_count=self.SAMPLE_COUNT,
                            suppress_count=self.SUPPRESS_COUNT,
                        ),
                        WeightedChoiceGenerator(
                            table_name=table_name,
                            column_name=column_name,
                            values=vg.cvs_not_suppressed,
                            counts=vg.counts_not_suppressed,
                            sample_count=self.SAMPLE_COUNT,
                            suppress_count=self.SUPPRESS_COUNT,
                        ),
                    ]
        return generators
