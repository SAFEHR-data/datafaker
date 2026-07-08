"""End-to-end tests for the MS-SQL Server dialect.

These tests require a running SQL Server instance.  Set the ``MSSQL_TEST_DSN``
environment variable to a ``mssql+pyodbc://`` connection string to enable them:

    export MSSQL_TEST_DSN="mssql+pyodbc://sa:Datafaker!Test123@localhost:1433/master?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"

With docker-compose:

    docker compose up -d mssql
    # wait ~30 s for SQL Server to start
"""
import asyncio
import os
from tempfile import mkstemp

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, text
from sqlalchemy import create_engine as sa_create_engine
from sqlalchemy.schema import CreateTable

import datafaker.create  # noqa: F401 — registers @compiles hooks (e.g. strip ON DELETE CASCADE)
from datafaker.db_utils import create_db_engine, create_db_engine_dst, get_sync_engine
from datafaker.make import make_src_stats, make_tables_file
from tests.utils import DatafakerTestCase, GeneratesDBTestCase, TestMSSQL

import yaml


# ---------------------------------------------------------------------------
# Instrument schema (mirrors tests/examples/instrument.sql in dialect-neutral
# SQLAlchemy — avoids the need for a T-SQL .sql fixture file).
# ---------------------------------------------------------------------------

def _make_src_metadata() -> MetaData:
    meta = MetaData()
    manufacturer = Table(
        "manufacturer",
        meta,
        Column("id", Integer, primary_key=True, autoincrement=False),
        Column("name", String(200), nullable=False),
        Column("founded", DateTime, nullable=False),
    )
    model = Table(
        "model",
        meta,
        Column("id", Integer, primary_key=True, autoincrement=False),
        Column("name", String(200), nullable=False),
        Column(
            "manufacturer_id",
            Integer,
            ForeignKey("manufacturer.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("introduced", DateTime, nullable=False),
    )
    string_table = Table(
        "string",
        meta,
        Column("id", Integer, primary_key=True, autoincrement=False),
        Column(
            "model_id",
            Integer,
            ForeignKey("model.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("position", Integer, nullable=False),
        Column("frequency", Float, nullable=False),
    )
    player = Table(
        "player",
        meta,
        Column("id", Integer, primary_key=True, autoincrement=False),
        Column("given_name", String(200), nullable=False),
        Column("family_name", String(200), nullable=False),
    )
    Table(
        "signature_model",
        meta,
        Column("id", Integer, primary_key=True, autoincrement=False),
        Column("name", String(20), nullable=False),
        Column("player_id", Integer, ForeignKey("player.id"), nullable=True),
        Column("based_on", Integer, ForeignKey("model.id"), nullable=True),
    )
    # suppress "unused variable" warnings
    _ = manufacturer, model, string_table, player
    return meta


_SRC_METADATA = _make_src_metadata()

_EXPECTED_TABLES = frozenset(
    {"manufacturer", "model", "string", "player", "signature_model"}
)


def _insert_sample_rows(engine) -> None:
    """Insert a minimal set of rows so make-stats has data to summarise."""
    with engine.begin() as conn:
        conn.execute(
            _SRC_METADATA.tables["manufacturer"].insert(),
            [
                {"id": 1, "name": "Blender", "founded": "1951-01-08 12:05:06"},
                {"id": 2, "name": "Gibbs", "founded": "1959-03-04 15:08:09"},
            ],
        )
        conn.execute(
            _SRC_METADATA.tables["model"].insert(),
            [
                {"id": 1, "name": "S-Type", "manufacturer_id": 1, "introduced": "1952-04-20 04:05:06"},
                {"id": 2, "name": "Pulse", "manufacturer_id": 1, "introduced": "1953-12-02 02:15:06"},
                {"id": 3, "name": "Paul Leslie", "manufacturer_id": 2, "introduced": "1960-02-20 04:05:06"},
            ],
        )
        conn.execute(
            _SRC_METADATA.tables["string"].insert(),
            [
                {"id": 1, "model_id": 1, "position": 1, "frequency": 329.6},
                {"id": 2, "model_id": 1, "position": 2, "frequency": 246.94},
                {"id": 3, "model_id": 2, "position": 1, "frequency": 98.0},
                {"id": 4, "model_id": 3, "position": 1, "frequency": 329.6},
            ],
        )
        conn.execute(
            _SRC_METADATA.tables["player"].insert(),
            [
                {"id": 1, "given_name": "Mark", "family_name": "Samson"},
                {"id": 2, "given_name": "Tim", "family_name": "Friedman"},
            ],
        )
        conn.execute(
            _SRC_METADATA.tables["signature_model"].insert(),
            [
                {"id": 1, "name": "Flame", "player_id": 1, "based_on": None},
                {"id": 2, "name": "Dragon", "player_id": None, "based_on": 1},
                {"id": 3, "name": "Veleno", "player_id": 2, "based_on": 2},
            ],
        )


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------


class MSSQLFunctionalTestCase(GeneratesDBTestCase):
    """End-to-end tests exercising the full datafaker pipeline against SQL Server."""

    database_type = TestMSSQL
    dump_file_path = None  # schema created in Python, not from a .sql file

    # Database names created inside SQL Server for this test run.
    database_name = "datafaker_test_src"
    _DST_DB = "dst"

    def setUp(self) -> None:
        """Create the source schema programmatically and wire up test-case fields.

        Bypasses RequiresDBTestCase.setUp (which calls run_sql) and reimplements
        the relevant parts so we can build the schema with SQLAlchemy MetaData.
        """
        # DatafakerTestCase.setUp handles CWD bookkeeping without touching the DB.
        DatafakerTestCase.setUp(self)

        TestMSSQL.setup()
        self.database = TestMSSQL()
        self.database.open()

        # Create the source database, build schema, and seed rows.
        self.database.create_empty(self.database_name, None)
        src_dsn = self.database.get_dsn(self.database_name)
        src_engine = sa_create_engine(src_dsn)
        _SRC_METADATA.create_all(src_engine)
        _insert_sample_rows(src_engine)
        src_engine.dispose()

        # Engine used by GeneratesDBTestCase helper methods (create_tables, etc.).
        self.engine = create_db_engine(src_dsn, schema_name=self.schema_name)
        self.sync_engine = get_sync_engine(self.engine)
        self.metadata = MetaData()
        self.metadata.reflect(self.sync_engine)

        # Empty destination database — create-tables will build the schema there.
        self.dst_database = TestMSSQL()
        self.dst_database.open()
        self.dst_database.create_empty(self._DST_DB, None)
        dst_dsn = self.dst_database.get_dsn(self.database_name)
        self.dst_name = self._DST_DB
        self.dst_metadata = MetaData()
        self.dst_engine = get_sync_engine(
            create_db_engine_dst(dst_dsn, schema_name=self.dst_schema_name)
        )

        # Write orm.yaml so generate_data() has the file it expects.
        (self.orm_fd, self.orm_file_path) = mkstemp(".yaml", "orm_", text=True)
        with os.fdopen(self.orm_fd, "w", encoding="utf-8") as fh:
            fh.write(make_tables_file(src_dsn, self.schema_name, engine=self.sync_engine))

        # Initialise stats/config path attributes expected by generate_data().
        self.stats_fd = 0
        self.stats_file_path = ""
        self.config_file_path = ""
        self.config_fd = 0

    def tearDown(self) -> None:
        # Dispose connection pools so the next setUp can drop these databases.
        if hasattr(self, "sync_engine"):
            self.sync_engine.dispose()
        if hasattr(self, "dst_engine") and self.dst_engine is not None:
            self.dst_engine.dispose()
        if self.database is not None:
            self.database.close()
        if self.dst_database is not None:
            self.dst_database.close()
        DatafakerTestCase.tearDown(self)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_smoke_connect(self) -> None:
        """ODBC driver can connect and run a trivial query."""
        engine = sa_create_engine(self.dsn)
        with engine.connect() as conn:
            row = conn.execute(text("SELECT 1 AS n")).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 1)

    def test_make_tables(self) -> None:
        """make_tables_file produces an orm.yaml listing the expected tables."""
        # setUp already called make_tables_file and wrote the result to orm_file_path
        with open(self.orm_file_path, encoding="utf-8") as fh:
            orm = yaml.safe_load(fh)
        # orm["tables"] is a dict keyed by table name
        table_names = set(orm.get("tables", {}).keys())
        self.assert_subset(_EXPECTED_TABLES, table_names)

    def test_make_stats(self) -> None:
        """make_src_stats runs without error against SQL Server.

        With an empty config there are no src-stats query blocks to run, so the
        function returns an empty dict — that is the correct behaviour.
        """
        loop = asyncio.new_event_loop()
        try:
            src_stats = loop.run_until_complete(
                make_src_stats(self.dsn, {}, self.schema_name)
            )
        finally:
            loop.close()
        self.assertIsInstance(src_stats, dict)

    def test_create_data(self) -> None:
        """Full pipeline: make-stats → create-tables → create-data inserts rows."""
        self.generate_data({})

        # Verify that at least the manufacturer table received rows.
        assert self.dst_engine is not None
        with self.dst_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM manufacturer")).scalar()
        self.assertGreater(count, 0, "Expected rows in manufacturer after create-data")

    def test_dialect_newid(self) -> None:
        """ChoiceProposer compiles its query with NEWID() not RANDOM() for mssql."""
        from sqlalchemy.dialects import mssql as mssql_dialect  # noqa: PLC0415
        from datafaker.proposers.choice import ZipfChoiceProposer  # noqa: PLC0415

        dialect = mssql_dialect.dialect()
        proposer = ZipfChoiceProposer(
            table_name="manufacturer",
            column_name="name",
            values=["Blender", "Gibbs"],
            counts=[5, 5],
            sample_count=2,
            dialect=dialect,
        )
        self.assertIn("newid()", proposer._query.lower())
        self.assertNotIn("random()", proposer._query.lower())

    def test_cascade_stripped(self) -> None:
        """The @compiles(CreateTable, 'mssql') hook strips ON DELETE CASCADE."""
        from sqlalchemy.dialects import mssql as mssql_dialect  # noqa: PLC0415

        model_table = _SRC_METADATA.tables["model"]
        ddl = str(CreateTable(model_table).compile(dialect=mssql_dialect.dialect()))
        self.assertIn("FOREIGN KEY", ddl)
        self.assertNotIn("ON DELETE CASCADE", ddl)
