"""Unit tests for proposers."""
import re

from datafaker.proposers.base import duckdb_workaround
from tests.utils import DatafakerTestCase

from sqlalchemy import dialects, select, Column, Table, Text, MetaData

select_re = re.compile(r"SELECT\s+([A-Za-z0-9_.\"]+)\s+FROM\s+([\"A-Za-z0-9_.]+)\s+AS\s+([\"A-Za-z0-9_.]+)", re.IGNORECASE)

class ProposerUnitTests(DatafakerTestCase):
    """Proposer test case."""

    def test_duckdb_workaround(self) -> None:
        """Test the duckdb_workaround function."""
        tabname = "tab1"
        colname = "col1"
        metadata = MetaData()
        table = Table(tabname, metadata)
        column = Column(colname, Text())
        table.append_column(column)
        stmt = select(column)
        stmt_a = duckdb_workaround(stmt)
        pgd = dialects.postgresql.dialect()
        sql = stmt_a.compile(dialect=pgd)
        grps = select_re.match(str(sql))
        self.assertIsNotNone(grps)
        tcs = grps.group(1).split(".")
        self.assertEqual(len(tcs), 2)
        self.assertEqual(tcs[0], grps.group(3))
        self.assertEqual(tcs[1], colname)
        self.assertEqual(grps.group(2), tabname)
