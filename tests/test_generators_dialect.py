"""Tests for dialect-correct SQL in generator classes."""
import unittest
import unittest.mock
from unittest.mock import MagicMock

from sqlalchemy import Column, Integer, MetaData, Table, literal_column, Dialect
from sqlalchemy.dialects import mssql, postgresql
from sqlalchemy.types import DateTime

from datafaker.dialects import SecondsDifference

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
        from datafaker.proposers.mimesis import MimesisDateTimeProposer

        column = self._make_column()
        engine = self._make_engine(postgresql.dialect)
        gens = MimesisDateTimeProposer.make_singleton(column, engine, "datetime.datetime")
        self.assertEqual(len(gens), 1)
        clauses = gens[0].select_aggregate_clauses()
        min_clause = clauses["birth_datetime__start"]["clause"]
        max_clause = clauses["birth_datetime__end"]["clause"]
        self.assertIn("EXTRACT", min_clause.upper())
        self.assertIn("EXTRACT", max_clause.upper())
        self.assertNotIn("DATEPART", min_clause.upper())

    def test_mssql_uses_datepart(self) -> None:
        """MS-SQL year clause uses DATEPART."""
        from datafaker.proposers.mimesis import MimesisDateTimeProposer

        column = self._make_column()
        engine = self._make_engine(mssql.dialect)
        gens = MimesisDateTimeProposer.make_singleton(column, engine, "datetime.datetime")
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
        from datafaker.proposers.base import Buckets

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
        tbl = Table("person", MetaData(), Column("age", Integer()))
        with unittest.mock.patch.object(Buckets, "__init__", return_value=None):
            Buckets.make_buckets(engine, tbl, tbl.c.age)

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
        self.assertIn("STDEVP(", sql)
        self.assertNotIn("STDDEV(", sql)  # function call form only, not the alias


class TestChoiceGeneratorStoredQuery(unittest.TestCase):
    """ChoiceGenerator._query is compiled to dialect-correct SQL at construction time."""

    def _make_gen(self, dialect, sample_count=None, suppress_count=0):
        from datafaker.proposers.choice import ZipfChoiceProposer
        return ZipfChoiceProposer(
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
        self.assertNotIn("RAND()", sql)
        self.assertNotIn(" TOP ", sql)
        self.assertNotIn("ROW_NUMBER()", sql)

    def test_mssql_sample_uses_rand_and_top(self) -> None:
        """MS-SQL stored query uses newid() and TOP for sampled path."""
        gen = self._make_gen(mssql.dialect(), sample_count=500)
        sql = gen._query.upper()
        self.assertIn("RAND()", sql)
        self.assertIn(" TOP ", sql)
        self.assertNotIn("RANDOM()", sql)
        self.assertNotIn("LIMIT", sql)

    def test_mssql_suppress_has_no_order_by(self) -> None:
        """MS-SQL suppress-only path emits no ORDER BY (was rejected without TOP)."""
        gen = self._make_gen(mssql.dialect(), suppress_count=7)
        sql = gen._query.upper()
        self.assertNotIn("ORDER BY", sql)

    def test_mssql_sample_and_suppress_uses_rand_and_top(self) -> None:
        """MS-SQL sample+suppress path uses newid()/TOP and no LIMIT/RANDOM."""
        gen = self._make_gen(mssql.dialect(), sample_count=500, suppress_count=7)
        sql = gen._query.upper()
        self.assertIn("RAND()", sql)
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
                self.assertNotIn("RAND()", sql)
                self.assertNotIn("NEWID()", sql)
                self.assertNotIn("LIMIT", sql)
                self.assertNotIn(" TOP ", sql)
                self.assertNotIn("ROW_NUMBER()", sql)


class TestChoiceGeneratorFactoryLiveQueries(unittest.TestCase):
    """ChoiceGeneratorFactory.get_generators executes dialect-correct live SQL."""

    def _captured_sqls(self, dialect, schema=None) -> list[str]:
        """Run get_proposers with a mocked engine and return compiled SQL strings."""
        from datafaker.proposers.choice import ChoiceProposerFactory

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
        ChoiceProposerFactory().get_proposers([tbl.c.gender], engine)

        return [
            str(s.compile(dialect=dialect, compile_kwargs={"literal_binds": True})).upper()
            for s in executed
        ]

    def test_mssql_live_queries_use_rand_and_top(self) -> None:
        """MS-SQL live queries use TOP (not LIMIT) and rand() (not random())."""
        sqls = self._captured_sqls(mssql.dialect())
        self.assertIn(" TOP ", sqls[0])
        self.assertNotIn("LIMIT", sqls[0])
        self.assertIn("RAND()", sqls[1])
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


class TestBucketsSchemaQualified(unittest.TestCase):
    """Buckets.make_buckets respects the schema of the src_table argument."""

    def _make_engine(self, dialect_name: str) -> MagicMock:
        engine = MagicMock()
        engine.dialect.name = dialect_name
        result = MagicMock()
        result.stddev = 5.0
        result.mean = 42.0
        result.configure_mock(**{"count": 100})
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        conn.execute.return_value.first.return_value = result
        conn.execute.return_value.__iter__ = MagicMock(return_value=iter([]))
        engine.connect.return_value = conn
        return engine

    def _get_make_buckets_sql(self, dialect_name: str, schema: str | None) -> str:
        from datafaker.proposers.base import Buckets

        engine = self._make_engine(dialect_name)
        meta = MetaData()
        tbl = Table("person", meta, Column("age", Integer()), schema=schema)

        executed_stmts = []
        orig_execute = engine.connect.return_value.execute

        def capture_execute(stmt, *args, **kwargs):
            executed_stmts.append(stmt)
            return orig_execute(stmt, *args, **kwargs)

        engine.connect.return_value.execute = capture_execute

        with unittest.mock.patch.object(Buckets, "__init__", return_value=None):
            Buckets.make_buckets(engine, tbl, tbl.c.age)

        self.assertGreaterEqual(len(executed_stmts), 1)
        dialect = mssql.dialect() if dialect_name == "mssql" else postgresql.dialect()
        return str(executed_stmts[0].compile(
            dialect=dialect,
            compile_kwargs={"literal_binds": True},
        )).upper()

    def test_schema_appears_in_from_mssql(self) -> None:
        """MS-SQL make_buckets query includes schema in FROM clause."""
        sql = self._get_make_buckets_sql("mssql", schema="myschema")
        self.assertIn("MYSCHEMA", sql)

    def test_schema_appears_in_from_postgresql(self) -> None:
        """PostgreSQL make_buckets query includes schema in FROM clause."""
        sql = self._get_make_buckets_sql("postgresql", schema="myschema")
        self.assertIn("MYSCHEMA", sql)

    def test_no_schema_omits_qualifier(self) -> None:
        """Without schema, FROM clause has no schema.table qualifier."""
        sql = self._get_make_buckets_sql("postgresql", schema=None)
        # A schema qualifier would appear as "SCHEMA.PERSON"; no dot before the table name.
        self.assertNotIn(".PERSON", sql)


class TestCovariateQueryDialect(unittest.TestCase):
    """CovariateQuery._inner_query() uses TOP/NEWID on MS-SQL and RANDOM/LIMIT elsewhere."""

    def _make_factory(self) -> MagicMock:
        factory = MagicMock()
        factory.query_predicate.return_value = ""
        return factory

    def _inner_query(self, dialect: Dialect) -> str:
        from datafaker.proposers.continuous import CovariateQuery

        metadata = MetaData()
        cq = (
            CovariateQuery(Table("person", metadata, Column("name")), self._make_factory(), dialect=dialect)
            .sample_count(500)
        )
        return cq._inner_query().upper()

    def test_mssql_uses_newid_and_top(self) -> None:
        """MS-SQL inner query uses SELECT TOP n … ORDER BY NEWID()."""
        sql = self._inner_query(mssql.dialect())
        self.assertIn(" TOP 500 ", sql)
        self.assertIn("NEWID()", sql)
        self.assertNotIn("RANDOM()", sql)
        self.assertNotIn("LIMIT", sql)

    def test_postgresql_uses_random_and_limit(self) -> None:
        """PostgreSQL inner query uses ORDER BY RANDOM() LIMIT n."""
        sql = self._inner_query(postgresql.dialect())
        self.assertIn("RANDOM()", sql)
        self.assertIn("LIMIT 500", sql)
        self.assertNotIn("NEWID()", sql)
        self.assertNotIn("TOP", sql)

    def test_no_sample_count_has_no_random_or_limit(self) -> None:
        """When sample_count is None no random ordering is emitted."""
        from datafaker.proposers.continuous import CovariateQuery

        for dialect in ("mssql", "postgresql", ""):
            with self.subTest(dialect=dialect):
                cq = CovariateQuery("person", self._make_factory(), dialect=dialect)
                sql = cq._inner_query().upper()
                self.assertNotIn("RANDOM()", sql)
                self.assertNotIn("NEWID()", sql)
                self.assertNotIn("LIMIT", sql)
                self.assertNotIn("TOP", sql)


class TestMissingnessQueryDialect(unittest.TestCase):
    """MissingnessType.sampled_query() produces dialect-correct SQL."""

    def setUp(self):
        super().setUp()
        self.metadata = MetaData()
        self.col_a = Column("col_a")
        self.col_b = Column("col_b")
        self.table = Table("person", self.metadata, self.col_a, self.col_b)

    def test_mssql_uses_rand_and_rownumber(self) -> None:
        """
        Test that MSSQL uses RAND and ROW_NUMBER for sampling.

        SELECT … ROW_NUMBER() AS MSSQL_RN
        WHERE MSSQL_RN < n ORDER BY RAND().
        """
        from datafaker.interactive.missingness import MissingnessType

        sql = MissingnessType.sampled_query(
            self.table, 1000, [self.col_a, self.col_b], dialect=mssql.dialect()
        ).upper()
        self.assertIn("ROW_NUMBER()", sql)
        self.assertIn("<= 1000", sql)
        self.assertIn("RAND()", sql)
        self.assertNotIn("RANDOM()", sql)
        self.assertNotIn("LIMIT", sql)

    def test_default_uses_random_and_limit(self) -> None:
        """Default (no dialect) sampled query uses RANDOM() and LIMIT."""
        from datafaker.interactive.missingness import MissingnessType

        sql = MissingnessType.sampled_query(self.table, 1000, [self.col_a], dialect=postgresql.dialect()).upper()
        self.assertIn("RANDOM()", sql)
        self.assertIn("LIMIT 1000", sql)
        self.assertNotIn("RAND()", sql)
        self.assertNotIn("NEWID()", sql)
        self.assertNotIn("TOP", sql)
        self.assertNotIn("ROW_NUMBER()", sql)

    def test_mssql_result_contains_column_null_checks(self) -> None:
        """MS-SQL sampled query retains IS NULL expressions for the named columns."""
        from datafaker.interactive.missingness import MissingnessType

        sql = MissingnessType.sampled_query(
            self.table, 500, [self.col_a], dialect=mssql.dialect()
        )
        self.assertIn("col_a IS NULL", sql)
        self.assertIn("col_a__is_null", sql)


class TestLogNormalGeneratorSchemaQualified(unittest.TestCase):
    """ContinuousLogDistributionGeneratorFactory respects src_table schema."""

    def _get_sql(self, schema: str | None) -> str:
        from datafaker.proposers.continuous import ContinuousLogDistributionProposerFactory

        meta = MetaData()
        tbl = Table("person", meta, Column("age", Integer()), schema=schema)

        executed_stmts = []
        result = MagicMock()
        result.logmean = 1.0
        result.logstddev = 0.5
        conn = MagicMock()
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        orig_execute = MagicMock(return_value=MagicMock(first=MagicMock(return_value=result)))

        def capture(stmt, *args, **kwargs):
            executed_stmts.append(stmt)
            return orig_execute(stmt, *args, **kwargs)

        conn.execute.side_effect = capture
        engine = MagicMock()
        engine.connect.return_value = conn

        from datafaker.proposers.base import Buckets
        import unittest.mock

        buckets = MagicMock(spec=Buckets)
        factory = ContinuousLogDistributionProposerFactory()
        with unittest.mock.patch.object(Buckets, "make_buckets", return_value=buckets):
            factory._get_generators_from_buckets(engine, tbl, tbl.c["age"], buckets)

        self.assertEqual(len(executed_stmts), 1)
        dialect = postgresql.dialect()
        return str(executed_stmts[0].compile(
            dialect=dialect,
            compile_kwargs={"literal_binds": True},
        )).upper()

    def test_schema_appears_in_from(self) -> None:
        """_get_generators_from_buckets includes schema in FROM clause."""
        sql = self._get_sql(schema="myschema")
        self.assertIn("MYSCHEMA", sql)

    def test_no_schema_omits_qualifier(self) -> None:
        """Without schema, FROM clause has no schema prefix."""
        sql = self._get_sql(schema=None)
        self.assertIn("FROM PERSON", sql)
        self.assertNotIn("FROM MYSCHEMA", sql)


class TestPredefinedGeneratorSchemaQualified(unittest.TestCase):
    """PredefinedGenerator parses aggregate clauses from schema-qualified SQL."""

    def _make_config(self, table_sql_name: str) -> dict:
        return {
            "tables": {
                "person": {
                    "row_generators": [
                        {
                            "name": "dist_gen.gaussian",
                            "columns_assigned": ["age"],
                            "kwargs": {
                                "mean": f'SRC_STATS["auto__person"]["results"][0]["mean__age"]',
                                "sd": f'SRC_STATS["auto__person"]["results"][0]["sd__age"]',
                            },
                        }
                    ]
                }
            },
            "src-stats": [
                {
                    "name": "auto__person",
                    "query": f"SELECT AVG(age) AS mean__age, STDDEV(age) AS sd__age FROM {table_sql_name}",
                    "comments": [],
                }
            ],
        }

    def test_unqualified_name_parses_clauses(self) -> None:
        """PredefinedProposer parses select_aggregate_clauses from unqualified FROM."""
        from datafaker.proposers.base import PredefinedProposer

        config = self._make_config("person")
        rg = config["tables"]["person"]["row_generators"][0]
        gen = PredefinedProposer("person", rg, config)
        self.assertIn("mean__age", gen.select_aggregate_clauses())
        self.assertIn("sd__age", gen.select_aggregate_clauses())

    def test_schema_qualified_name_parses_clauses(self) -> None:
        """PredefinedProposer parses select_aggregate_clauses from schema-qualified FROM."""
        from datafaker.proposers.base import PredefinedProposer

        config = self._make_config("myschema.person")
        rg = config["tables"]["person"]["row_generators"][0]
        gen = PredefinedProposer("person", rg, config)
        self.assertIn("mean__age", gen.select_aggregate_clauses())
        self.assertIn("sd__age", gen.select_aggregate_clauses())


class TestContinuousStddevDialect(unittest.TestCase):
    """ContinuousDistributionProposer and LogNormalProposer emit STDEV on MSSQL."""

    def _make_table(self) -> tuple:
        meta = MetaData()
        tbl = Table("person", meta, Column("age", Integer()))
        return tbl, tbl.c.age

    def test_gaussian_postgresql_uses_stddev(self) -> None:
        """GaussianProposer.select_aggregate_clauses uses STDDEV on PostgreSQL."""
        from datafaker.proposers.continuous import GaussianProposer

        tbl, col = self._make_table()
        proposer = GaussianProposer(tbl, col, MagicMock(), dialect=postgresql.dialect())
        clause = proposer.select_aggregate_clauses()["stddev__age"]["clause"]
        self.assertIn("STDDEV", clause.upper())

    def test_gaussian_mssql_uses_stdev(self) -> None:
        """GaussianProposer.select_aggregate_clauses uses STDEV on MSSQL."""
        from datafaker.proposers.continuous import GaussianProposer

        tbl, col = self._make_table()
        proposer = GaussianProposer(tbl, col, MagicMock(), dialect=mssql.dialect())
        clause = proposer.select_aggregate_clauses()["stddev__age"]["clause"]
        self.assertIn("STDEVP", clause.upper())
        self.assertNotIn("STDDEV", clause.upper())

    def test_lognormal_postgresql_uses_stddev(self) -> None:
        """LogNormalProposer.select_aggregate_clauses uses STDDEV on PostgreSQL."""
        from datafaker.proposers.continuous import LogNormalProposer

        tbl, col = self._make_table()
        proposer = LogNormalProposer(tbl, col, MagicMock(), 1.0, 0.5, dialect=postgresql.dialect())
        clause = proposer.select_aggregate_clauses()["logstddev__age"]["clause"]
        self.assertIn("STDDEV", clause.upper())

    def test_lognormal_mssql_uses_stdev(self) -> None:
        """LogNormalProposer.select_aggregate_clauses uses STDEV on MSSQL."""
        from datafaker.proposers.continuous import LogNormalProposer

        tbl, col = self._make_table()
        proposer = LogNormalProposer(tbl, col, MagicMock(), 1.0, 0.5, dialect=mssql.dialect())
        clause = proposer.select_aggregate_clauses()["logstddev__age"]["clause"]
        self.assertIn("STDEVP", clause.upper())
        self.assertNotIn("STDDEV", clause.upper())


class TestIntervalsDifferenceDialect(unittest.TestCase):
    """SecondsDifference compiles to DATEDIFF on MSSQL and EXTRACT(EPOCH) on PostgreSQL."""

    def _make_element(self):

        return SecondsDifference(literal_column("t1"), literal_column("t2"))

    def test_postgresql_uses_extract_epoch(self) -> None:
        """PostgreSQL SecondsDifference uses EXTRACT(EPOCH FROM …)."""
        elem = self._make_element()
        sql = str(elem.compile(dialect=postgresql.dialect())).upper()
        self.assertIn("EXTRACT", sql)
        self.assertIn("EPOCH", sql)
        self.assertNotIn("DATEDIFF", sql)

    def test_mssql_uses_datediff(self) -> None:
        """MSSQL SecondsDifference uses DATEDIFF(second, …)."""
        elem = self._make_element()
        sql = str(elem.compile(dialect=mssql.dialect())).upper()
        self.assertIn("DATEDIFF", sql)
        self.assertNotIn("EXTRACT", sql)
        self.assertNotIn("EPOCH", sql)

    def test_date_after_proposer_stddev_clause_mssql(self) -> None:
        """DateAfterProposer.select_aggregate_clauses uses STDEV and DATEDIFF on MSSQL."""
        from sqlalchemy import Column
        from sqlalchemy.types import DateTime
        from datafaker.proposers.intervals import DateAfterProposer

        meta = MetaData()
        tbl = Table("visit", meta, Column("start_date", DateTime()), Column("end_date", DateTime()))
        proposer = DateAfterProposer(
            metadata=meta,
            sd=1.0,
            mean=86400.0,
            column=tbl.c.end_date,
            anchor=tbl.c.start_date,
            dialect=mssql.dialect(),
        )
        clauses = proposer.select_aggregate_clauses()
        mean_clause = clauses["mean__end_date"]["clause"].upper()
        sd_clause = clauses["stddev__end_date"]["clause"].upper()
        self.assertIn("DATEDIFF", mean_clause)
        self.assertIn("DATEDIFF", sd_clause)
        self.assertIn("STDEV", sd_clause)
        self.assertNotIn("STDDEV", sd_clause)

    def test_date_after_proposer_stddev_clause_postgresql(self) -> None:
        """DateAfterProposer.select_aggregate_clauses uses STDDEV and EXTRACT on PostgreSQL."""
        from sqlalchemy import Column
        from sqlalchemy.types import DateTime
        from datafaker.proposers.intervals import DateAfterProposer

        meta = MetaData()
        tbl = Table("visit", meta, Column("start_date", DateTime()), Column("end_date", DateTime()))
        proposer = DateAfterProposer(
            metadata=meta,
            sd=1.0,
            mean=86400.0,
            column=tbl.c.end_date,
            anchor=tbl.c.start_date,
            dialect=postgresql.dialect(),
        )
        clauses = proposer.select_aggregate_clauses()
        mean_clause = clauses["mean__end_date"]["clause"].upper()
        sd_clause = clauses["stddev__end_date"]["clause"].upper()
        self.assertIn("EXTRACT", mean_clause)
        self.assertIn("EXTRACT", sd_clause)
        self.assertIn("STDDEV", sd_clause)


class TestAggregateQuerySchemaQualified(unittest.TestCase):
    """_get_aggregate_query qualifies table names using the engine's schema_translate_map."""

    def _make_engine(self, schema: str | None) -> MagicMock:
        engine = MagicMock()
        schema_map = {None: schema} if schema else {}
        engine.get_execution_options.return_value = {"schema_translate_map": schema_map}
        return engine

    def _make_gen(self) -> MagicMock:
        from datafaker.proposers.base import Proposer

        gen = MagicMock(spec=Proposer)
        gen.select_aggregate_clauses.return_value = {
            "mean__age": {"clause": "AVG(age)", "comment": None}
        }
        return gen

    def test_aggregate_query_includes_schema(self) -> None:
        """get_aggregate_query qualifies the bare table name when engine has a schema map."""
        from datafaker.interactive.generators import get_aggregate_query

        engine = self._make_engine("myschema")
        gen = self._make_gen()
        result = get_aggregate_query([gen], "person", engine)
        self.assertIsNotNone(result)
        self.assertIn("myschema.person", result)

    def test_aggregate_query_no_schema(self) -> None:
        """get_aggregate_query uses the bare name when no schema is set."""
        from datafaker.interactive.generators import get_aggregate_query

        engine = self._make_engine(None)
        gen = self._make_gen()
        result = get_aggregate_query([gen], "person", engine)
        self.assertIsNotNone(result)
        self.assertIn("person", result)
        # No schema qualifier (schema.table) should appear after FROM
        self.assertNotIn(".", result.split("FROM ")[-1].strip('"').strip())
