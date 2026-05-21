"""Tests for dialect-correct SQL in generator classes."""
import unittest
import unittest.mock
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
        sql = self._get_executed_sql("mssql")
        self.assertIn("STDEV(", sql)
        self.assertNotIn("STDDEV(", sql)  # function call form only, not the alias


class TestChoiceGeneratorStoredQuery(unittest.TestCase):
    """ChoiceGenerator._query is compiled to dialect-correct SQL at construction time."""

    def _make_gen(self, dialect, sample_count=None, suppress_count=0):
        from datafaker.generators.choice import ZipfChoiceGenerator
        return ZipfChoiceGenerator(
            table_name="patient",
            column_name="gender",
            values=["M", "F"],
            counts=[70, 30],
            sample_count=sample_count,
            suppress_count=suppress_count,
            dialect=dialect,
        )

    def test_postgresql_sample_uses_random_and_limit(self) -> None:
        """PostgreSQL stored query uses random() and LIMIT for sampled path."""
        gen = self._make_gen(postgresql.dialect(), sample_count=500)
        sql = gen._query.upper()
        self.assertIn("RANDOM()", sql)
        self.assertIn("LIMIT", sql)
        self.assertNotIn("NEWID()", sql)
        self.assertNotIn(" TOP ", sql)

    def test_mssql_sample_uses_newid_and_top(self) -> None:
        """MS-SQL stored query uses newid() and TOP for sampled path."""
        gen = self._make_gen(mssql.dialect(), sample_count=500)
        sql = gen._query.upper()
        self.assertIn("NEWID()", sql)
        self.assertIn(" TOP ", sql)
        self.assertNotIn("RANDOM()", sql)
        self.assertNotIn("LIMIT", sql)

    def test_mssql_suppress_has_no_order_by(self) -> None:
        """MS-SQL suppress-only path emits no ORDER BY (was rejected without TOP)."""
        gen = self._make_gen(mssql.dialect(), suppress_count=7)
        sql = gen._query.upper()
        self.assertNotIn("ORDER BY", sql)

    def test_mssql_sample_and_suppress_uses_newid_and_top(self) -> None:
        """MS-SQL sample+suppress path uses newid()/TOP and no LIMIT/RANDOM."""
        gen = self._make_gen(mssql.dialect(), sample_count=500, suppress_count=7)
        sql = gen._query.upper()
        self.assertIn("NEWID()", sql)
        self.assertIn(" TOP ", sql)
        self.assertNotIn("RANDOM()", sql)
        self.assertNotIn("LIMIT", sql)

    def test_no_sample_no_suppress_has_no_random_or_limit(self) -> None:
        """No-sample path never includes RANDOM/LIMIT regardless of dialect."""
        for dialect in (postgresql.dialect(), mssql.dialect()):
            with self.subTest(dialect=dialect.name):
                gen = self._make_gen(dialect)
                sql = gen._query.upper()
                self.assertNotIn("RANDOM()", sql)
                self.assertNotIn("NEWID()", sql)
                self.assertNotIn("LIMIT", sql)
                self.assertNotIn(" TOP ", sql)


class TestChoiceGeneratorFactoryLiveQueries(unittest.TestCase):
    """ChoiceGeneratorFactory.get_generators executes dialect-correct live SQL."""

    def _captured_sqls(self, dialect, schema=None) -> list[str]:
        """Run get_generators with a mocked engine and return compiled SQL strings."""
        from datafaker.generators.choice import ChoiceGeneratorFactory

        engine = MagicMock()
        engine.dialect = dialect

        row_count = MagicMock()
        row_count.v = "M"
        row_count.f = 70
        result_count = MagicMock()
        result_count.rowcount = 1
        result_count.__iter__ = MagicMock(return_value=iter([row_count]))

        row_sample = MagicMock()
        row_sample.v = "M"
        row_sample.f = 70
        result_sample = MagicMock()
        result_sample.__iter__ = MagicMock(return_value=iter([row_sample]))

        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        engine.connect.return_value = conn

        executed = []
        results_queue = [result_count, result_sample]

        def capture(stmt, *args, **kwargs):
            executed.append(stmt)
            return results_queue[len(executed) - 1]

        conn.execute.side_effect = capture

        meta = MetaData()
        tbl = Table("patient", meta, Column("gender", Integer()), schema=schema)
        ChoiceGeneratorFactory().get_generators([tbl.c.gender], engine)

        return [
            str(s.compile(dialect=dialect, compile_kwargs={"literal_binds": True})).upper()
            for s in executed
        ]

    def test_mssql_live_queries_use_top_and_newid(self) -> None:
        """MS-SQL live queries use TOP (not LIMIT) and newid() (not random())."""
        sqls = self._captured_sqls(mssql.dialect())
        self.assertIn(" TOP ", sqls[0])
        self.assertNotIn("LIMIT", sqls[0])
        self.assertIn(" TOP ", sqls[1])
        self.assertIn("NEWID()", sqls[1])
        self.assertNotIn("LIMIT", sqls[1])
        self.assertNotIn("RANDOM()", sqls[1])

    def test_postgresql_live_queries_use_limit_and_random(self) -> None:
        """PostgreSQL live queries use LIMIT and random()."""
        sqls = self._captured_sqls(postgresql.dialect())
        self.assertIn("LIMIT", sqls[0])
        self.assertNotIn(" TOP ", sqls[0])
        self.assertIn("LIMIT", sqls[1])
        self.assertIn("RANDOM()", sqls[1])
        self.assertNotIn(" TOP ", sqls[1])
        self.assertNotIn("NEWID()", sqls[1])

    def test_schema_qualified_table_appears_in_from(self) -> None:
        """Schema-qualified table name is included in the FROM clause on both dialects."""
        for dialect in (mssql.dialect(), postgresql.dialect()):
            with self.subTest(dialect=dialect.name):
                sqls = self._captured_sqls(dialect, schema="myschema")
                for sql in sqls:
                    self.assertIn("MYSCHEMA", sql)
