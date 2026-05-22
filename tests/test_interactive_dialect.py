"""Tests for dialect-correct SQL in interactive shell methods."""
import unittest
from unittest.mock import MagicMock

from sqlalchemy.dialects import mssql, postgresql


def _make_engine(dialect) -> MagicMock:
    """Return a mock engine whose dialect is the given SQLAlchemy dialect instance."""
    engine = MagicMock()
    engine.dialect = dialect
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    executed = []

    def capture(stmt, *args, **kwargs):
        executed.append(stmt)
        result = MagicMock()
        result.keys.return_value = []
        result.fetchmany.return_value = []
        result.all.return_value = []
        return result

    conn.execute.side_effect = capture
    engine.connect.return_value = conn
    engine._executed = executed
    return engine


def _compiled(stmt, dialect) -> str:
    return str(stmt.compile(dialect=dialect, compile_kwargs={"literal_binds": True})).upper()


class TestPeekDialect(unittest.TestCase):
    """DbCmd.do_peek() uses NEWID/TOP on MS-SQL and RANDOM/LIMIT on PostgreSQL."""

    def _run_peek(self, dialect_instance, col_names=None):
        from datafaker.interactive.base import DbCmd

        engine = _make_engine(dialect_instance)
        shell = MagicMock(spec=DbCmd)
        shell.sync_engine = engine
        shell.table_index = 0
        shell._table_entries = [MagicMock()]
        shell.table_name.return_value = "person"
        shell._get_column_names.return_value = col_names or ["gender_concept_id"]
        shell.print_table = MagicMock()
        shell.print = MagicMock()

        DbCmd.do_peek(shell, " ".join(col_names) if col_names else "")
        return engine._executed, dialect_instance

    def test_mssql_peek_uses_newid_and_top(self) -> None:
        """MS-SQL do_peek compiles to TOP … NEWID()."""
        executed, dialect = self._run_peek(mssql.dialect())
        self.assertEqual(len(executed), 1)
        sql = _compiled(executed[0], dialect)
        self.assertIn("TOP", sql)
        self.assertIn("NEWID()", sql)
        self.assertNotIn("LIMIT", sql)
        self.assertNotIn("RANDOM()", sql)

    def test_postgresql_peek_uses_random_and_limit(self) -> None:
        """PostgreSQL do_peek compiles to RANDOM() … LIMIT."""
        executed, dialect = self._run_peek(postgresql.dialect())
        self.assertEqual(len(executed), 1)
        sql = _compiled(executed[0], dialect)
        self.assertIn("RANDOM()", sql)
        self.assertIn("LIMIT", sql)
        self.assertNotIn("NEWID()", sql)
        self.assertNotIn(" TOP ", sql)


class TestGetColumnDataDialect(unittest.TestCase):
    """GeneratorCmd._get_column_data() uses NEWID/TOP on MS-SQL and RANDOM/LIMIT on PostgreSQL."""

    def _run_get_column_data(self, dialect_instance):
        from datafaker.interactive.generators import GeneratorCmd

        engine = _make_engine(dialect_instance)
        shell = MagicMock(spec=GeneratorCmd)
        shell.sync_engine = engine
        shell.table_name.return_value = "person"
        shell._get_column_names.return_value = ["gender_concept_id"]

        GeneratorCmd._get_column_data(shell, 5)
        return engine._executed, dialect_instance

    def test_mssql_uses_newid_and_top(self) -> None:
        """MS-SQL _get_column_data compiles to TOP … NEWID()."""
        executed, dialect = self._run_get_column_data(mssql.dialect())
        self.assertEqual(len(executed), 1)
        sql = _compiled(executed[0], dialect)
        self.assertIn("TOP", sql)
        self.assertIn("NEWID()", sql)
        self.assertNotIn("LIMIT", sql)
        self.assertNotIn("RANDOM()", sql)

    def test_postgresql_uses_random_and_limit(self) -> None:
        """PostgreSQL _get_column_data compiles to RANDOM() … LIMIT."""
        executed, dialect = self._run_get_column_data(postgresql.dialect())
        self.assertEqual(len(executed), 1)
        sql = _compiled(executed[0], dialect)
        self.assertIn("RANDOM()", sql)
        self.assertIn("LIMIT", sql)
        self.assertNotIn("NEWID()", sql)
        self.assertNotIn(" TOP ", sql)


class TestPrintColumnDataDialect(unittest.TestCase):
    """TableCmd.print_column_data() uses NEWID/TOP on MS-SQL and RANDOM/LIMIT on PostgreSQL."""

    def _run_print_column_data(self, dialect_instance):
        from datafaker.interactive.table import TableCmd

        engine = _make_engine(dialect_instance)
        shell = MagicMock(spec=TableCmd)
        shell.sync_engine = engine
        shell.table_name.return_value = "person"
        shell.columnize = MagicMock()

        TableCmd.print_column_data(shell, "gender_concept_id", 10, 0)
        return engine._executed, dialect_instance

    def test_mssql_uses_newid_and_top(self) -> None:
        """MS-SQL print_column_data compiles to TOP … NEWID()."""
        executed, dialect = self._run_print_column_data(mssql.dialect())
        self.assertEqual(len(executed), 1)
        sql = _compiled(executed[0], dialect)
        self.assertIn("TOP", sql)
        self.assertIn("NEWID()", sql)
        self.assertNotIn("LIMIT", sql)
        self.assertNotIn("RANDOM()", sql)

    def test_postgresql_uses_random_and_limit(self) -> None:
        """PostgreSQL print_column_data compiles to RANDOM() … LIMIT."""
        executed, dialect = self._run_print_column_data(postgresql.dialect())
        self.assertEqual(len(executed), 1)
        sql = _compiled(executed[0], dialect)
        self.assertIn("RANDOM()", sql)
        self.assertIn("LIMIT", sql)
        self.assertNotIn("NEWID()", sql)
        self.assertNotIn(" TOP ", sql)
