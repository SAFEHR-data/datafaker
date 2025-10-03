"""Base table generator classes."""
import functools
import gzip
import math
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sqlalchemy import Connection, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import Table

from datafaker.utils import (
    MAKE_VOCAB_PROGRESS_REPORT_EVERY,
    logger,
    stream_yaml,
    table_row_count,
)


@functools.cache
def zipf_weights(size):
    total = sum(map(lambda n: 1 / n, range(1, size + 1)))
    return [1 / (n * total) for n in range(1, size + 1)]


def merge_with_constants(xs: list, constants_at: dict[int, any]):
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
    for x in xs[xi:]:
        yield x


class NothingToGenerateException(Exception):
    def __init__(self, message):
        super().__init__(message)


class DistributionGenerator:
    root3 = math.sqrt(3)

    def __init__(self):
        self.np_gen = np.random.default_rng()

    def uniform(self, low, high) -> float:
        return random.uniform(float(low), float(high))

    def uniform_ms(self, mean, sd) -> float:
        m = float(mean)
        h = self.root3 * float(sd)
        return random.uniform(m - h, m + h)

    def normal(self, mean, sd) -> float:
        return random.normalvariate(float(mean), float(sd))

    def lognormal(self, logmean, logsd) -> float:
        return random.lognormvariate(float(logmean), float(logsd))

    def choice(self, a):
        c = random.choice(a)
        return c["value"] if type(c) is dict and "value" in c else c

    def zipf_choice(self, a, n=None):
        if n is None:
            n = len(a)
        c = random.choices(a, weights=zipf_weights(n))[0]
        return c["value"] if type(c) is dict and "value" in c else c

    def weighted_choice(self, a: list[dict[str, any]]) -> list[any]:
        """
        Choice weighted by the count in the original dataset.
        :param a: a list of dicts, each with a ``value`` key
        holding the value to be returned and a ``count`` key holding the
        number of that value found in the original dataset
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

    def constant(self, value):
        return value

    def multivariate_normal_np(self, cov):
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

    def _select_group(self, alts: list[dict[str, any]]):
        """
        Choose one of the ``alts`` weighted by their ``"count"`` elements.
        """
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
        raise Exception("Internal error: ran out of choices in _select_group")

    def _find_constants(self, result: dict[str, any]):
        """
        Find all keys ``kN``, returning a dictionary of ``N: kNN``.

        This can be passed into ``merge_with_constants`` as the
        ``constants_at`` argument.
        """
        out: dict[int, any] = {}
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

    def multivariate_normal(self, cov):
        """
        Produce a list of values pulled from a multivariate distribution.

        :param cov: A dict with various keys: ``rank`` is the number of
        output values, ``m0``, ``m1``, ... are the means of the
        distributions (``rank`` of them). ``c0_0``, ``c0_1``, ``c1_1``, ...
        are the covariates, ``cN_M`` is the covariate of the ``N``th and
        ``M``th varaibles, with 0 <= ``N`` <= ``M`` < ``rank``.
        :return: list of ``rank`` floating point values
        """
        return self.multivariate_normal_np(cov).tolist()

    def multivariate_lognormal(self, cov):
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
        return np.exp(self.multivariate_normal_np(cov)).tolist()

    def grouped_multivariate_normal(self, covs):
        cov = self._select_group(covs)
        logger.debug("Multivariate normal group selected: %s", cov)
        constants = self._find_constants(cov)
        nums = self.multivariate_normal(cov)
        return list(merge_with_constants(nums, constants))

    def grouped_multivariate_lognormal(self, covs):
        cov = self._select_group(covs)
        logger.debug("Multivariate lognormal group selected: %s", cov)
        constants = self._find_constants(cov)
        nums = np.exp(self.multivariate_normal_np(cov)).tolist()
        return list(merge_with_constants(nums, constants))

    def _check_generator_name(self, name: str) -> None:
        if name not in self.PERMITTED_SUBGENS:
            raise Exception("%s is not a permitted generator", name)

    def alternatives(
        self, alternative_configs: list[dict[str, any]], counts: list[int] | None
    ):
        """
        A generator that picks between other generators.

        :param alternative_configs: List of alternative generators.
        Each alternative has the following keys: "count" -- a weight for
        how often to use this alternative; "name" -- which generator
        for this partition, for example "composite"; "params" -- the
        parameters for this alternative.
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
        self, constants_at: list[int], subgen: str, params: dict[str, any]
    ):
        if subgen not in self.PERMITTED_SUBGENS:
            logger.error(
                "subgenerator %s is not a valid name. Valid names are %s.",
                subgen,
                self.PERMITTED_SUBGENS,
            )
        subout = getattr(self, subgen)(**params)
        logger.debug("Merging constants %s", constants_at)
        return list(merge_with_constants(subout, constants_at))

    def truncated_string(self, subgen_fn, params, length):
        """Calls ``subgen_fn(**params)`` and truncates the results to ``length``."""
        result = subgen_fn(**params)
        if result is None:
            return None
        return result[:length]


class TableGenerator(ABC):
    """Abstract base class for table generator classes."""

    num_rows_per_pass: int = 1

    @abstractmethod
    def __call__(self, dst_db_conn: Connection) -> dict[str, Any]:
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
            opener = lambda: open(yaml_file, mode="r", encoding="utf-8")
        else:
            yaml_file = base_path / Path(self.table.fullname + ".yaml.gz")
            if yaml_file.exists():
                opener = lambda: gzip.open(yaml_file, mode="rt")
            else:
                logger.warning("File %s not found. Skipping...", yaml_file)
                return
        if 0 < table_row_count(self.table, connection):
            logger.warning(
                "Table %s already contains data (consider running 'datafaker remove-vocab'), skipping...",
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
    def sampled(self, patterns):
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
