"""Tests for dialect-correct SQL in generator classes."""
import unittest
from unittest.mock import MagicMock

from sqlalchemy import Column, Integer, MetaData, Table
from sqlalchemy.dialects import mssql, postgresql
from sqlalchemy.types import DateTime


class TestMimesisDateTimeDialect(unittest.TestCase):
    """MimesisDateTimeGenerator.make_singleton compiles year expressions per dialect."""

    def _make_column(self) -> Column:
        meta = MetaData()
        t = Table("person", meta, Column("birth_datetime", DateTime()))
        return t.c.birth_datetime

    def _make_engine(self, dialect) -> MagicMock:
        engine = MagicMock()
        engine.dialect = dialect()
        result = MagicMock()
        result.start = 1950
        result.end = 2000
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.execute.return_value.first.return_value = result
        engine.connect.return_value = conn
        return engine

    def test_postgresql_uses_extract(self) -> None:
        """PostgreSQL year clause uses EXTRACT."""
        from datafaker.generators.mimesis import MimesisDateTimeGenerator

        column = self._make_column()
        engine = self._make_engine(postgresql.dialect)
        gens = MimesisDateTimeGenerator.make_singleton(column, engine, "datetime.datetime")
        self.assertEqual(len(gens), 1)
        clauses = gens[0].select_aggregate_clauses()
        min_clause = clauses["birth_datetime__start"]["clause"]
        max_clause = clauses["birth_datetime__end"]["clause"]
        self.assertIn("EXTRACT", min_clause.upper())
        self.assertIn("EXTRACT", max_clause.upper())
        self.assertNotIn("DATEPART", min_clause.upper())

    def test_mssql_uses_datepart(self) -> None:
        """MS-SQL year clause uses DATEPART."""
        from datafaker.generators.mimesis import MimesisDateTimeGenerator

        column = self._make_column()
        engine = self._make_engine(mssql.dialect)
        gens = MimesisDateTimeGenerator.make_singleton(column, engine, "datetime.datetime")
        self.assertEqual(len(gens), 1)
        clauses = gens[0].select_aggregate_clauses()
        min_clause = clauses["birth_datetime__start"]["clause"]
        max_clause = clauses["birth_datetime__end"]["clause"]
        self.assertIn("DATEPART", min_clause.upper())
        self.assertIn("DATEPART", max_clause.upper())
        self.assertNotIn("EXTRACT", min_clause.upper())


class TestBucketsStddevDialect(unittest.TestCase):
    """Buckets.make_buckets uses STDEV on MS-SQL and STDDEV on other dialects."""

    def _make_engine_with_dialect_name(self, dialect_name: str) -> MagicMock:
        engine = MagicMock()
        engine.dialect.name = dialect_name
        result = MagicMock()
        result.stddev = 5.0
        result.mean = 42.0
        # count attribute via getattr
        result.configure_mock(**{"count": 100})
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.execute.return_value.first.return_value = result
        engine.connect.return_value = conn
        return engine

    def _get_executed_sql(self, dialect_name: str) -> str:
        from datafaker.generators.base import Buckets

        engine = self._make_engine_with_dialect_name(dialect_name)
        # make_buckets will call engine.connect().execute(stmt)
        # We patch it to capture the compiled SQL
        executed_stmts = []
        orig_execute = engine.connect.return_value.execute

        def capture_execute(stmt, *args, **kwargs):
            executed_stmts.append(stmt)
            return orig_execute(stmt, *args, **kwargs)

        engine.connect.return_value.execute = capture_execute
        # Prevent the Buckets constructor from running (it uses a separate query)
        with unittest.mock.patch.object(Buckets, "__init__", return_value=None):
            Buckets.make_buckets(engine, "person", "age")

        self.assertEqual(len(executed_stmts), 1)
        compiled = str(executed_stmts[0].compile(
            dialect=mssql.dialect() if dialect_name == "mssql" else postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        ))
        return compiled.upper()

    def test_postgresql_uses_stddev(self) -> None:
        """PostgreSQL query uses STDDEV function."""
        import unittest.mock
        sql = self._get_executed_sql("postgresql")
        self.assertIn("STDDEV(", sql)  # function call form

    def test_mssql_uses_stdev(self) -> None:
        """MS-SQL query uses STDEV function (no trailing D)."""
        import unittest.mock
        sql = self._get_executed_sql("mssql")
        self.assertIn("STDEV(", sql)
        self.assertNotIn("STDDEV(", sql)  # function call form only, not the alias
