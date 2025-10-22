"""Base table generator classes."""
import functools
import gzip
import math
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Generator

import numpy as np
import yaml
from sqlalchemy import Connection, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import MetaData, Table

from datafaker.utils import (
    MAKE_VOCAB_PROGRESS_REPORT_EVERY,
    T,
    logger,
    stream_yaml,
    table_row_count,
)


class InappropriateGeneratorException(Exception):
    """Exception thrown if a generator is requested that is not appropriate."""


@functools.cache
def zipf_weights(size: int) -> list[float]:
    """Get the weights of a Zipf distribution of a given size."""
    total = sum(map(lambda n: 1 / n, range(1, size + 1)))
    return [1 / (n * total) for n in range(1, size + 1)]


def merge_with_constants(
    xs: list[T], constants_at: dict[int, T]
) -> Generator[T, None, None]:
    """
    Merge a list of items with other items that must be placed at certain indices.

    :param constants_at: A map of indices to objects that must be placed at
    those indices.
    :param xs: Items that fill in the gaps left by ``constants_at``.
    :return: ``xs`` with ``constants_at`` inserted at the appropriate
    points. If there are not enough elements in ``xs`` to fill in the gaps
    in ``constants_at``, the elements of ``constants_at`` after the gap
    are dropped.
    """
    outi = 0
    xi = 0
    constant_count = len(constants_at)
    while constant_count != 0:
        if outi in constants_at:
            yield constants_at[outi]
            constant_count -= 1
        else:
            if xi == len(xs):
                return
            yield xs[xi]
            xi += 1
        outi += 1
    yield from xs[xi:]


class NothingToGenerateException(Exception):
    """Exception thrown when no value can be generated."""

    def __init__(self, message: str):
        """Initialise the exception with a human-readable message."""
        super().__init__(message)


class DistributionGenerator:
    """An object that can produce values from various distributions."""

    root3 = math.sqrt(3)

    def __init__(self) -> None:
        """Initialise the DistributionGenerator."""
        self.np_gen = np.random.default_rng()

    def uniform(self, low: float, high: float) -> float:
        """
        Choose a value according to a uniform distribution.

        :param low: The lowest value that can be chosen.
        :param high: The highest value that can be chosen.
        :return: The output value.
        """
        return random.uniform(float(low), float(high))

    def uniform_ms(self, mean: float, sd: float) -> float:
        """
        Choose a value according to a uniform distribution.

        :param mean: The mean of the output values.
        :param sd: The standard deviation of the output values.
        :return: The output value.
        """
        m = float(mean)
        h = self.root3 * float(sd)
        return random.uniform(m - h, m + h)

    def normal(self, mean: float, sd: float) -> float:
        """
        Choose a value according to a Gaussian (normal) distribution.

        :param mean: The mean of the output values.
        :param sd: The standard deviation of the output values.
        :return: The output value.
        """
        return random.normalvariate(float(mean), float(sd))

    def lognormal(self, logmean: float, logsd: float) -> float:
        """
        Choose a value according to a lognormal distribution.

        :param logmean: The mean of the logs of the output values.
        :param logsd: The standard deviation of the logs of the output values.
        :return: The output value.
        """
        return random.lognormvariate(float(logmean), float(logsd))

    def choice_direct(self, a: list[T]) -> T:
        """
        Choose a value with equal probability.

        :param a: The list of values to output.
        :return: The chosen value.
        """
        return random.choice(a)

    def choice(self, a: list[Mapping[str, T]]) -> T | None:
        """
        Choose a value with equal probability.

        :param a: The list of values to output. Each element is a mapping with
        a key ``value`` and the key is the value to return.
        :return: The chosen value.
        """
        return self.choice_direct(a).get("value", None)

    def zipf_choice_direct(self, a: list[T], n: int | None = None) -> T:
        """
        Choose a value according to the Zipf distribution.

        The nth value (starting from 1) is chosen with a frequency
        1/n times as frequently as the first value is chosen.

        :param a: The list of values to output, most frequent first.
        :return: The chosen value.
        """
        if n is None:
            n = len(a)
        return random.choices(a, weights=zipf_weights(n))[0]

    def zipf_choice(self, a: list[Mapping[str, T]], n: int | None = None) -> T | None:
        """
        Choose a value according to the Zipf distribution.

        The nth value (starting from 1) is chosen with a frequency
        1/n times as frequently as the first value is chosen.

        :param a: The list of rows to choose between, most frequent first.
        Each element is a mapping with a key ``value`` and the key is the
        value to return.
        :return: The chosen value.
        """
        c = self.zipf_choice_direct(a, n)
        return c.get("value", None)

    def weighted_choice(self, a: list[dict[str, Any]]) -> Any:
        """
        Choice weighted by the count in the original dataset.

        :param a: a list of dicts, each with a ``value`` key
        holding the value to be returned and a ``count`` key holding the
        number of that value found in the original dataset
        :return: The chosen ``value``.
        """
        vs = []
        counts = []
        for vc in a:
            count = vc.get("count", 0)
            if count:
                counts.append(count)
                vs.append(vc.get("value", None))
        c = random.choices(vs, weights=counts)[0]
        return c

    def constant(self, value: T) -> T:
        """Return the same value always."""
        return value

    def multivariate_normal_np(self, cov: dict[str, Any]) -> np.typing.NDArray:
        """
        Return an array of values chosen from the given covariates.

        :param cov: Keys are ``rank``: The number of values to output;
        ``mN``: The mean of variable ``N`` (where ``N`` is between 0 and
        one less than ``rank``). ``cN_M`` (where 0 < ``N`` <= ``M`` < ``rank``):
        the covariance between the ``N``th and the ``M``th variables.
        :return: A numpy array of results.
        """
        rank = int(cov["rank"])
        if rank == 0:
            return np.empty(shape=(0,))
        mean = [float(cov[f"m{i}"]) for i in range(rank)]
        covs = [
            [
                float(cov[f"c{i}_{j}"] if i <= j else cov[f"c{j}_{i}"])
                for i in range(rank)
            ]
            for j in range(rank)
        ]
        return self.np_gen.multivariate_normal(mean, covs)

    def _select_group(self, alts: list[dict[str, Any]]) -> Any:
        """Choose one of the ``alts`` weighted by their ``"count"`` elements."""
        total = 0
        for alt in alts:
            if alt["count"] < 0:
                logger.warning(
                    "Alternative count is %d, but should not be negative", alt["count"]
                )
            else:
                total += alt["count"]
        if total == 0:
            raise NothingToGenerateException("No counts in any alternative")
        choice = random.randrange(total)
        for alt in alts:
            choice -= alt["count"]
            if choice < 0:
                return alt
        raise NothingToGenerateException(
            "Internal error: ran out of choices in _select_group"
        )

    def _find_constants(self, result: dict[str, Any]) -> dict[int, Any]:
        """
        Find all keys ``kN``, returning a dictionary of ``N: kNN``.

        This can be passed into ``merge_with_constants`` as the
        ``constants_at`` argument.
        """
        out: dict[int, Any] = {}
        for k, v in result.items():
            if k.startswith("k") and k[1:].isnumeric():
                out[int(k[1:])] = v
        return out

    PERMITTED_SUBGENS = {
        "multivariate_lognormal",
        "multivariate_normal",
        "grouped_multivariate_lognormal",
        "grouped_multivariate_normal",
        "constant",
        "weighted_choice",
        "with_constants_at",
    }

    def multivariate_normal(self, cov: dict[str, Any]) -> list[float]:
        """
        Produce a list of values pulled from a multivariate distribution.

        :param cov: A dict with various keys: ``rank`` is the number of
        output values, ``m0``, ``m1``, ... are the means of the
        distributions (``rank`` of them). ``c0_0``, ``c0_1``, ``c1_1``, ...
        are the covariates, ``cN_M`` is the covariate of the ``N``th and
        ``M``th varaibles, with 0 <= ``N`` <= ``M`` < ``rank``.
        :return: list of ``rank`` floating point values
        """
        out: list[float] = self.multivariate_normal_np(cov).tolist()
        return out

    def multivariate_lognormal(self, cov: dict[str, Any]) -> list[float]:
        """
        Produce a list of values pulled from a multivariate distribution.

        :param cov: A dict with various keys: ``rank`` is the number of
        output values, ``m0``, ``m1``, ... are the means of the
        distributions (``rank`` of them). ``c0_0``, ``c0_1``, ``c1_1``, ...
        are the covariates, ``cN_M`` is the covariate of the ``N``th and
        ``M``th varaibles, with 0 <= ``N`` <= ``M`` < ``rank``. These
        are all the means and covariants of the logs of the data.
        :return: list of ``rank`` floating point values
        """
        out: list[Any] = np.exp(self.multivariate_normal_np(cov)).tolist()
        return out

    def grouped_multivariate_normal(self, covs: list[dict[str, Any]]) -> list[Any]:
        """Produce a list of values pulled from a set of multivariate distributions."""
        cov = self._select_group(covs)
        logger.debug("Multivariate normal group selected: %s", cov)
        constants = self._find_constants(cov)
        nums = self.multivariate_normal(cov)
        return list(merge_with_constants(nums, constants))

    def grouped_multivariate_lognormal(self, covs: list[dict[str, Any]]) -> list[Any]:
        """Produce a list of values pulled from a set of multivariate distributions."""
        cov = self._select_group(covs)
        logger.debug("Multivariate lognormal group selected: %s", cov)
        constants = self._find_constants(cov)
        nums = np.exp(self.multivariate_normal_np(cov)).tolist()
        return list(merge_with_constants(nums, constants))

    def _check_generator_name(self, name: str) -> None:
        if name not in self.PERMITTED_SUBGENS:
            raise InappropriateGeneratorException(
                f"{name} is not a permitted generator"
            )

    def alternatives(
        self,
        alternative_configs: list[dict[str, Any]],
        counts: list[dict[str, int]] | None,
    ) -> Any:
        """
        Pick between other generators.

        :param alternative_configs: List of alternative generators.
        Each alternative has the following keys: "count" -- a weight for
        how often to use this alternative; "name" -- which generator
        for this partition, for example "composite"; "params" -- the
        parameters for this alternative.
        :param counts: A list of weights for each alternative. If None, the
        "count" value of each alternative is used. Each count is a dict
        with a "count" key.
        :return: list of values
        """
        if counts is not None:
            while True:
                count = self._select_group(counts)
                alt = alternative_configs[count["index"]]
                name = alt["name"]
                self._check_generator_name(name)
                try:
                    return getattr(self, name)(**alt["params"])
                except NothingToGenerateException:
                    # Prevent this alternative from being chosen again
                    count["count"] = 0
        alt = self._select_group(alternative_configs)
        name = alt["name"]
        self._check_generator_name(name)
        return getattr(self, name)(**alt["params"])

    def with_constants_at(
        self, constants_at: dict[int, T], subgen: str, params: dict[str, T]
    ) -> list[T]:
        """
        Insert constants into the results of a different generator.

        :param constants_at: A dictionary of positions and objects to insert
        into the return list at those positions.
        :param subgen: The name of the function to call to get the results
        that will have the constants inserted into.
        :param params: Keyword arguments to the ``subgen`` function.
        :return: A list of results from calling ``subgen(**params)``
        with ``constants_at`` inserted in at the appropriate indices.
        """
        if subgen not in self.PERMITTED_SUBGENS:
            logger.error(
                "subgenerator %s is not a valid name. Valid names are %s.",
                subgen,
                self.PERMITTED_SUBGENS,
            )
        subout = getattr(self, subgen)(**params)
        logger.debug("Merging constants %s", constants_at)
        return list(merge_with_constants(subout, constants_at))

    def truncated_string(
        self, subgen_fn: Callable[..., list[T]], params: dict, length: int
    ) -> list[T]:
        """Call ``subgen_fn(**params)`` and truncate the results to ``length``."""
        result = subgen_fn(**params)
        if result is None:
            return None
        return result[:length]


class TableGenerator(ABC):
    """Abstract base class for table generator classes."""

    num_rows_per_pass: int = 1

    @abstractmethod
    def __call__(self, dst_db_conn: Connection, metadata: MetaData) -> dict[str, Any]:
        """Return, as a dictionary, a new row for the table that we are generating.

        The only argument, `dst_db_conn`, should be a database connection to the
        database to which the data is being written. Most generators won't use it, but
        some do, and thus it's required by the interface.

        The return value should be a dictionary with column names as strings for keys,
        and the values being the values for the new row.
        """


@dataclass
class FileUploader:
    """For uploading data files."""

    table: Table

    def _load_existing_file(
        self, connection: Connection, file_size: int, opener: Callable[[], Any]
    ) -> None:
        count = 0
        with opener() as fh:
            rows = stream_yaml(fh)
            for row in rows:
                stmt = insert(self.table).values(row)
                connection.execute(stmt)
                connection.commit()
                count += 1
                if count % MAKE_VOCAB_PROGRESS_REPORT_EVERY == 0:
                    logger.info(
                        "inserted row %d of %s, %.1f%%",
                        count,
                        self.table.name,
                        100 * fh.tell() / file_size,
                    )

    def load(self, connection: Connection, base_path: Path = Path(".")) -> None:
        """Load the data from file."""
        yaml_file = base_path / Path(self.table.fullname + ".yaml")
        if yaml_file.exists():

            def opener() -> TextIOWrapper:
                return open(yaml_file, mode="r", encoding="utf-8")

        else:
            yaml_file = base_path / Path(self.table.fullname + ".yaml.gz")
            if yaml_file.exists():

                def opener() -> TextIOWrapper:
                    return gzip.open(yaml_file, mode="rt")

            else:
                logger.warning("File %s not found. Skipping...", yaml_file)
                return
        if 0 < table_row_count(self.table, connection):
            logger.warning(
                (
                    "Table %s already contains data"
                    " (consider running 'datafaker remove-vocab'), skipping..."
                ),
                self.table.name,
            )
            return
        try:
            file_size = os.path.getsize(yaml_file)
            self._load_existing_file(connection, file_size, opener)
        except yaml.YAMLError as e:
            logger.warning("Error reading YAML file %s: %s", yaml_file, e)
            return
        except SQLAlchemyError as e:
            logger.warning(
                "Error inserting rows into table %s: %s", self.table.fullname, e
            )


class ColumnPresence:
    """Object for generators to use for missingness completely at random."""

    def sampled(self, patterns: list[dict[str, Any]]) -> set[str]:
        """
        Select a random pattern and output the non-null columns.

        :param patterns: List of outputs from missingness SQL queries.
        Columns in each output: ``row_count`` is the number of rows
        with this missingness pattern, then for each column
        ``<column>`` there is a boolean called ``missingness__is_null``.
        :return: All the names of the columns no make non-null.
        """
        total = 0
        for pattern in patterns:
            total += pattern.get("row_count", 0)
        s = random.randrange(total)
        for pattern in patterns:
            s -= pattern.get("row_count", 0)
            if s < 0:
                cs = set()
                for column, nullness in pattern.items():
                    if not nullness and column.endswith("__is_null"):
                        cs.add(column[:-9])
                return cs
        logger.error("failed to sample patterns")
        return set()
