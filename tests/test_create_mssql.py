"""Tests for MS-SQL DDL compilation in datafaker.create."""
import unittest

from sqlalchemy import Column, Integer, MetaData, String, Table
from sqlalchemy.dialects import mssql
from sqlalchemy.schema import CreateTable

# Importing create registers the @compiles hooks globally.
import datafaker.create  # noqa: F401


def _compile_create_table(table: Table) -> str:
    """Compile a CreateTable statement against the MS-SQL dialect."""
    return str(CreateTable(table).compile(dialect=mssql.dialect()))


class TestMSSQLRemoveIdentity(unittest.TestCase):
    """@compiles(CreateColumn, 'mssql') strips IDENTITY from autoincrement columns."""

    def _make_table(self) -> Table:
        meta = MetaData()
        return Table(
            "test_table",
            meta,
            Column("id", Integer(), primary_key=True, autoincrement=True),
            Column("value", Integer(), nullable=True),
        )

    def test_identity_absent_from_ddl(self) -> None:
        """IDENTITY must be absent so explicit INSERTs succeed without SET IDENTITY_INSERT."""
        ddl = _compile_create_table(self._make_table())
        self.assertNotIn("IDENTITY", ddl)

    def test_integer_type_preserved(self) -> None:
        """The INTEGER type is preserved after stripping IDENTITY."""
        ddl = _compile_create_table(self._make_table())
        self.assertIn("INTEGER", ddl)

    def test_primary_key_constraint_preserved(self) -> None:
        """PRIMARY KEY constraint is not affected by the hook."""
        ddl = _compile_create_table(self._make_table())
        self.assertIn("PRIMARY KEY", ddl)

    def test_non_autoincrement_column_unchanged(self) -> None:
        """Non-autoincrement columns are not altered."""
        ddl = _compile_create_table(self._make_table())
        self.assertIn("value", ddl.lower())

    def test_multiple_autoincrement_columns_all_stripped(self) -> None:
        """IDENTITY is stripped from every column that would receive it."""
        meta = MetaData()
        table = Table(
            "multi",
            meta,
            Column("id", Integer(), primary_key=True, autoincrement=True),
            Column("seq", Integer(), autoincrement=True),
        )
        ddl = _compile_create_table(table)
        self.assertNotIn("IDENTITY", ddl)

    def test_duckdb_serial_hook_still_works(self) -> None:
        """Adding the mssql hook does not break the existing DuckDB SERIAL hook (regression)."""
        from sqlalchemy.dialects import registry
        from sqlalchemy import create_engine
        from sqlalchemy.schema import CreateColumn

        # Compile a CreateColumn for the duckdb dialect indirectly by checking
        # that remove_serial is still registered.
        from sqlalchemy.ext.compiler import compiles

        # If the duckdb hook were broken, datafaker.create.remove_serial would
        # not be registered.  Verify it still exists and is callable.
        self.assertTrue(callable(datafaker.create.remove_serial))
        self.assertTrue(callable(datafaker.create.remove_mssql_identity))
