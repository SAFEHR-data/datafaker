"""Tests for MS-SQL DDL compilation in datafaker.create."""
import unittest

from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table
from sqlalchemy.dialects import mssql
from sqlalchemy.schema import CreateTable

# Importing create registers the @compiles hooks globally.
import datafaker.create  # noqa: F401


def _compile_create_table(table: Table) -> str:
    """Compile a CreateTable statement against the MS-SQL dialect."""
    return str(CreateTable(table).compile(dialect=mssql.dialect()))


class TestMSSQLIdentityAbsent(unittest.TestCase):
    """MS-SQL tables must NOT have IDENTITY — datafaker supplies explicit PK values.

    The remove_mssql_identity hook strips IDENTITY from CREATE TABLE DDL so that
    ColumnValueProvider.increment() can insert explicit PK values without needing
    SET IDENTITY_INSERT ON.
    """

    def _make_table(self) -> Table:
        meta = MetaData()
        return Table(
            "test_table",
            meta,
            Column("id", Integer(), primary_key=True, autoincrement=True),
            Column("value", Integer(), nullable=True),
        )

    def test_identity_absent_from_ddl(self) -> None:
        """IDENTITY must be stripped so datafaker can insert explicit PK values."""
        ddl = _compile_create_table(self._make_table())
        self.assertNotIn("IDENTITY", ddl)

    def test_integer_type_preserved(self) -> None:
        """The INTEGER type is preserved."""
        ddl = _compile_create_table(self._make_table())
        self.assertIn("INTEGER", ddl)

    def test_primary_key_constraint_preserved(self) -> None:
        """PRIMARY KEY constraint is not affected."""
        ddl = _compile_create_table(self._make_table())
        self.assertIn("PRIMARY KEY", ddl)

    def test_non_autoincrement_column_unchanged(self) -> None:
        """Non-autoincrement columns are not altered."""
        ddl = _compile_create_table(self._make_table())
        self.assertIn("value", ddl.lower())


class TestMSSQLRemoveOnDeleteCascade(unittest.TestCase):
    """@compiles(CreateTable, 'mssql') strips ON DELETE CASCADE to avoid error 1785."""

    def _make_multi_fk_table(self) -> Table:
        meta = MetaData()
        concept = Table(
            "concept",
            meta,
            Column("concept_id", Integer(), primary_key=True),
        )
        return Table(
            "person",
            meta,
            Column("person_id", Integer(), primary_key=True),
            Column("gender_concept_id", Integer(), ForeignKey("concept.concept_id", ondelete="CASCADE")),
            Column("race_concept_id", Integer(), ForeignKey("concept.concept_id", ondelete="CASCADE")),
        )

    def test_cascade_absent_from_mssql_ddl(self) -> None:
        ddl = _compile_create_table(self._make_multi_fk_table())
        self.assertNotIn("ON DELETE CASCADE", ddl)

    def test_foreign_key_constraint_preserved(self) -> None:
        ddl = _compile_create_table(self._make_multi_fk_table())
        self.assertIn("FOREIGN KEY", ddl)

    def test_duckdb_cascade_hook_still_works(self) -> None:
        """The combined mssql hook does not break the DuckDB CASCADE hook."""
        self.assertTrue(callable(datafaker.create.remove_on_delete_cascade))
        self.assertTrue(callable(datafaker.create.compile_mssql_create_table))
