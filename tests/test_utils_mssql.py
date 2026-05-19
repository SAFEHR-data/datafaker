"""Tests for MS-SQL driver support helpers in datafaker.utils."""
import sys
import unittest
from unittest.mock import MagicMock, call, patch


class TestMakeAsyncDsn(unittest.TestCase):
    """Tests for make_async_dsn."""

    def _call(self, dsn: str) -> str:
        from datafaker.utils import make_async_dsn

        return make_async_dsn(dsn)

    def test_postgresql_bare_dialect(self) -> None:
        """postgresql:// is rewritten to use asyncpg."""
        result = self._call("postgresql://user:pass@host:5432/db")
        self.assertTrue(
            result.startswith("postgresql+asyncpg://"),
            f"Expected asyncpg driver, got: {result}",
        )

    def test_postgresql_with_existing_driver(self) -> None:
        """postgresql+psycopg2:// is also rewritten to asyncpg."""
        result = self._call("postgresql+psycopg2://user:pass@host:5432/db")
        self.assertTrue(result.startswith("postgresql+asyncpg://"))

    def test_postgresql_preserves_credentials_and_path(self) -> None:
        """Host, port and database name are preserved (password is masked in repr)."""
        from sqlalchemy.engine import make_url

        result_url = make_url(self._call("postgresql://alice:secret@dbhost:5433/mydb"))
        self.assertEqual(result_url.host, "dbhost")
        self.assertEqual(result_url.port, 5433)
        self.assertEqual(result_url.database, "mydb")
        self.assertEqual(result_url.username, "alice")

    def test_mssql_bare_dialect(self) -> None:
        """mssql:// is rewritten to use aioodbc."""
        result = self._call("mssql://user:pass@host:1433/db")
        self.assertTrue(
            result.startswith("mssql+aioodbc://"),
            f"Expected aioodbc driver, got: {result}",
        )

    def test_mssql_with_existing_driver(self) -> None:
        """mssql+pyodbc:// is rewritten to aioodbc."""
        result = self._call("mssql+pyodbc://user:pass@host:1433/db")
        self.assertTrue(result.startswith("mssql+aioodbc://"))

    def test_unknown_dialect_raises(self) -> None:
        """An unknown dialect raises ValueError rather than silently producing a bad DSN."""
        with self.assertRaises(ValueError) as ctx:
            self._call("oracle://user:pass@host:1521/db")
        self.assertIn("oracle", str(ctx.exception))

    def test_duckdb_raises(self) -> None:
        """DuckDB DSNs are not async-capable and should raise."""
        with self.assertRaises(ValueError):
            self._call("duckdb:///path/to/file.db")


class TestIsUndefinedObjectError(unittest.TestCase):
    """Tests for _is_undefined_object_error."""

    def _call(self, exc: Exception) -> bool:
        from datafaker.utils import _is_undefined_object_error

        return _is_undefined_object_error(exc)

    def test_pgcode_42704_returns_true(self) -> None:
        """Any exception with pgcode 42704 is treated as UndefinedObject."""
        exc = Exception("undefined object")
        exc.pgcode = "42704"  # type: ignore[attr-defined]
        self.assertTrue(self._call(exc))

    def test_other_pgcode_returns_false(self) -> None:
        """Exceptions with a different pgcode are not matched."""
        exc = Exception("some other error")
        exc.pgcode = "23505"  # type: ignore[attr-defined]
        self.assertFalse(self._call(exc))

    def test_no_pgcode_returns_false(self) -> None:
        """Exceptions without a pgcode attribute are not matched."""
        self.assertFalse(self._call(ValueError("no pgcode here")))

    def test_psycopg2_undefined_object_returns_true(self) -> None:
        """A real psycopg2 UndefinedObject exception is matched (if psycopg2 installed)."""
        try:
            import psycopg2.errors  # type: ignore[import]
        except ImportError:
            self.skipTest("psycopg2 not installed")

        exc = psycopg2.errors.UndefinedObject("constraint does not exist")
        self.assertTrue(self._call(exc))

    def test_works_without_psycopg2(self) -> None:
        """_is_undefined_object_error falls back to pgcode check when psycopg2 is absent."""
        with patch.dict(sys.modules, {"psycopg2": None, "psycopg2.errors": None}):
            exc = Exception("constraint does not exist")
            exc.pgcode = "42704"  # type: ignore[attr-defined]
            from datafaker.utils import _is_undefined_object_error

            self.assertTrue(_is_undefined_object_error(exc))

    def test_import_does_not_require_psycopg2(self) -> None:
        """datafaker.utils can be imported even when psycopg2 is unavailable."""
        with patch.dict(sys.modules, {"psycopg2": None, "psycopg2.errors": None}):
            import importlib

            import datafaker.utils as utils_mod

            importlib.reload(utils_mod)

    def test_pyodbc_style_error_without_pgcode_returns_false(self) -> None:
        """A pyodbc-style error has SQLSTATE in args[0] but no pgcode attribute.

        pyodbc does not set pgcode, so the current implementation cannot
        distinguish a 'constraint does not exist' pyodbc error from any other
        exception without pgcode.  This test documents that known limitation.
        """
        exc = Exception("constraint does not exist")
        # pyodbc puts SQLSTATE in args[0]; MS-SQL error 3728 maps to SQLSTATE 42000
        exc.args = ("42000", "[42000] [SQL Server] ... is not a constraint. (3728)")
        self.assertFalse(self._call(exc))

    def test_pgcode_none_returns_false(self) -> None:
        """pgcode=None is not treated as a match."""
        exc = Exception("undefined object")
        exc.pgcode = None  # type: ignore[attr-defined]
        self.assertFalse(self._call(exc))

    def test_sqlalchemy_wrapper_not_matched_only_orig_is(self) -> None:
        """The helper expects the unwrapped DBAPI error (e.orig), not the SQLAlchemy wrapper.

        The call site in utils.py passes e.orig, not e, so the SQLAlchemy
        ProgrammingError itself should not match even when e.orig would.
        """
        from sqlalchemy.exc import ProgrammingError

        orig = Exception("underlying DBAPI error")
        orig.pgcode = "42704"  # type: ignore[attr-defined]
        sa_exc = ProgrammingError("statement", {}, orig)

        # The SQLAlchemy exception itself has no pgcode.
        self.assertFalse(self._call(sa_exc))
        # Passing e.orig directly does match.
        self.assertTrue(self._call(sa_exc.orig))


class TestSchemaTranslateMap(unittest.TestCase):
    """Tests for the cross-dialect schema routing in create_db_engine."""

    def _make_engine(self, dsn: str, schema_name: str | None = None):
        from datafaker.utils import create_db_engine, get_sync_engine

        return get_sync_engine(create_db_engine(dsn, schema_name=schema_name))

    def test_no_schema_no_translate_map(self) -> None:
        """Without a schema_name, schema_translate_map is absent from execution options."""
        engine = self._make_engine("duckdb:///:memory:")
        opts = engine.get_execution_options()
        self.assertNotIn("schema_translate_map", opts)

    def test_schema_sets_translate_map(self) -> None:
        """When schema_name is given, schema_translate_map routes None to that schema."""
        engine = self._make_engine("duckdb:///:memory:", schema_name="myschema")
        opts = engine.get_execution_options()
        self.assertIn("schema_translate_map", opts)
        self.assertEqual(opts["schema_translate_map"], {None: "myschema"})

    def test_search_path_not_issued(self) -> None:
        """set_db_settings is never called with a search_path key."""
        from datafaker.utils import create_db_engine, get_sync_engine

        with patch("datafaker.utils.set_db_settings") as mock_set:
            engine = get_sync_engine(
                create_db_engine("duckdb:///:memory:", schema_name="myschema")
            )
            # Force a connection so any connect-event handler would fire
            with engine.connect() as conn:
                conn.execute(__import__("sqlalchemy").text("SELECT 1"))

            for c in mock_set.call_args_list:
                settings_arg = c.args[1] if len(c.args) > 1 else c.kwargs.get("settings", {})
                self.assertNotIn(
                    "search_path",
                    settings_arg,
                    "search_path must not be passed to set_db_settings",
                )

    def test_mssql_dsn_schema_sets_translate_map(self) -> None:
        """schema_translate_map is set even for an MS-SQL DSN (engine creation, no connect)."""
        from datafaker.utils import create_db_engine, get_sync_engine

        # create_engine with mssql+pyodbc does not connect at construction time,
        # so this is safe to run even without an ODBC driver installed.
        try:
            engine = get_sync_engine(
                create_db_engine("mssql+pyodbc://user:pass@host/db", schema_name="dbo")
            )
        except Exception:
            self.skipTest("mssql+pyodbc driver not available in this environment")

        opts = engine.get_execution_options()
        self.assertEqual(opts.get("schema_translate_map"), {None: "dbo"})


class TestGetMetadataSchema(unittest.TestCase):
    """Tests for the schema_name parameter on get_metadata."""

    def test_reflect_called_with_schema(self) -> None:
        """get_metadata passes schema_name to MetaData.reflect."""
        from datafaker.utils import get_metadata

        mock_engine = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        with patch("datafaker.utils.MetaData") as MockMetaData:
            mock_md = MagicMock()
            MockMetaData.return_value = mock_md
            mock_md.reflect.return_value = None

            get_metadata(mock_engine, schema_name="myschema")

            mock_md.reflect.assert_called_once_with(mock_engine, schema="myschema")

    def test_reflect_called_without_schema_when_none(self) -> None:
        """get_metadata passes schema=None to reflect when no schema_name is given."""
        from datafaker.utils import get_metadata

        mock_engine = MagicMock()

        with patch("datafaker.utils.MetaData") as MockMetaData:
            mock_md = MagicMock()
            MockMetaData.return_value = mock_md
            mock_md.reflect.return_value = None

            get_metadata(mock_engine)

            mock_md.reflect.assert_called_once_with(mock_engine, schema=None)
