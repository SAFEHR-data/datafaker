"""Data dumping functions."""
import csv
import io
from typing import TYPE_CHECKING

import sqlalchemy
from sqlalchemy.schema import MetaData

from datafaker.utils import create_db_engine, get_sync_engine, logger

if TYPE_CHECKING:
    from _csv import Writer


def _make_csv_writer(file: io.TextIOBase) -> "Writer":
    """Make the standard CSV file writer."""
    return csv.writer(file, quoting=csv.QUOTE_MINIMAL)


def dump_db_tables(
    metadata: MetaData,
    dsn: str,
    schema: str | None,
    table_name: str,
    file: io.TextIOBase,
) -> None:
    """Output the table as CSV."""
    if table_name not in metadata.tables:
        logger.error("%s is not a table described in the ORM file", table_name)
        return
    table = metadata.tables[table_name]
    csv_out = _make_csv_writer(file)
    csv_out.writerow(table.columns.keys())
    engine = get_sync_engine(create_db_engine(dsn, schema_name=schema))
    with engine.connect() as connection:
        result = connection.execute(sqlalchemy.select(table))
        for row in result:
            csv_out.writerow(row)
