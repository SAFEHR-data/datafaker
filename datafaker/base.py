"""Base table generator classes."""
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
import functools
import math
import numpy as np
import os
from pathlib import Path
import random
from typing import Any

import yaml
import gzip
from sqlalchemy import Connection, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import Table

from datafaker.utils import (
    logger,
    stream_yaml,
    MAKE_VOCAB_PROGRESS_REPORT_EVERY,
    table_row_count,
)

@functools.cache
def zipf_weights(size):
    total = sum(map(lambda n: 1/n, range(1, size + 1)))
    return [
        1 / (n * total)
        for n in range(1, size + 1)
    ]


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
        :param a: a list of dicts, each with a ``v`` key
        holding the value to be returned and a ``count`` key holding the
        number of that value found in the original dataset
        """
        vs = []
        counts = []
        for vc in a:
            count = vc.get("count", 0)
            if count:
                counts.append(count)
                vs.append(vc.get("v", None))
        c = random.choices(vs, weights=counts)[0]
        return c

    def constant(self, value):
        return value

    def multivariate_normal_np(self, cov):
        rank = int(cov["rank"])
        mean = [
            float(cov[f"m{i}"])
            for i in range(rank)
        ]
        covs = [
            [
                float(cov[f"c{i}_{j}"] if i <= j else cov[f"c{j}_{i}"])
                for i in range(rank)
            ]
            for j in range(rank)
        ]
        return self.np_gen.multivariate_normal(mean, covs)

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

    def _load_existing_file(self, connection: Connection, file_size: int, opener: Callable[[], Any]) -> None:
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

    def load(self, connection: Connection, base_path: Path=Path(".")) -> None:
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
            logger.warning("Table %s already contains data (consider running 'datafaker remove-vocab'), skipping...", self.table.name)
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
