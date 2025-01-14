"""Base table generator classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy
import os
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import Connection, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import Table

from sqlsynthgen.utils import (
    logger,
    stream_yaml,
    MAKE_VOCAB_PROGRESS_REPORT_EVERY,
)

def zipf_weights(size):
    total = sum(map(lambda n: 1/n, range(1, size + 1)))
    return [
        1 / (n * total)
        for n in range(1, size + 1)
    ]


class DistributionGenerator:
    def __init__(self):
        self.rng = numpy.random.default_rng()

    def uniform(self, low: float, high: float) -> float:
        return self.rng.uniform(low=low, high=high)

    def normal(self, mean: float, sd: float) -> float:
        return self.rng.normal(loc=mean, scale=sd)

    def choice(self, a):
        return self.rng.choice(a).item()

    def zipf_choice(self, a, n):
        return self.rng.choice(a, p = zipf_weights(n)).item()


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

    def load(self, connection: Connection) -> None:
        """Load the data from file."""
        yaml_file = Path(self.table.fullname + ".yaml")
        if not yaml_file.exists():
            logger.warning("File %s not found. Skipping...", yaml_file)
            return
        try:
            file_size = os.path.getsize(yaml_file)
            count = 0
            with open(yaml_file, "r", encoding="utf-8") as fh:
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
        except yaml.YAMLError as e:
            logger.warning("Error reading YAML file %s: %s", yaml_file, e)
            return
        except SQLAlchemyError as e:
            logger.warning(
                "Error inserting rows into table %s: %s", self.table.fullname, e
            )
