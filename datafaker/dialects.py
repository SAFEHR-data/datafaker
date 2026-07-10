"""Dialect differences."""
from collections.abc import Mapping
import re
from typing import Any

from sqlalchemy import Column, Select, Table, Join
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import CreateSchema, CreateTable
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import NamedFromClause
from sqlalchemy.sql.visitors import (
    ExternallyTraversible,
    InternalTraversal,
    replacement_traverse,
    traverse,
)
from sqlalchemy.types import Date, DateTime

serial_re = re.compile(r"\bSERIAL\b")


class TableReplacer:
    """
    Replaces tables with aliased tables.

    We need this to work around a DuckDB problem:
    If we are using the ORM code to select a column ``c`` from a table
    ``t.parquet``, then DuckDB expects the SQL
    ``SELECT "t.parquet".c FROM "t.parquet"`` if ``t.parquet`` is an actual
    table in the database, or ``SELECT t.c FROM "t.parquet"`` if ``t.parquet``
    names a file. The best way around this seems to be to use an aliased table,
    which works in both cases: ``SELECT a.c FROM "t.parquet" AS a``, and the
    best way for that to happen seems to be to use ``replacement_traverse``.
    """

    def __init__(self, table: Table) -> None:
        """Initialise with the table to be aliased."""
        self.table = table
        self.atable = self.table.alias(f"_{table.name}__alias")

    def replace(
        self, obj: ExternallyTraversible, **_kw: Any
    ) -> ExternallyTraversible | None:
        """Replace columns with the same column on the aliased table."""
        if isinstance(obj, Column):
            if obj.table == self.table:
                return self.atable.columns[obj.name]
        elif isinstance(obj, Table) and obj == self.table:
            return self.atable
        elif isinstance(obj, NamedFromClause):
            # Return the same object rather than None
            # to supress descent into this object
            return obj
        return None

    def aliased_table(self) -> NamedFromClause:
        """Get the aliased table."""
        return self.atable


@compiles(Select, "duckdb")
def duckdb_workaround(element: Select, compiler: Any, **kw: Any) -> Any:
    """
    Transform a SQLAlchemy ORM statement to work around DuckDB issues.

    :param stmt: An ORM statement, such as the return value of ``select``.
    :return: An ORM statement, transformed if necessary.
    """
    #sel =  compiler.visit_select(element, **kw)
    #breakpoint()
    #return sel
    tables: set[Table] = set()
    traverse(element, {}, {"table": tables.add})
    for t in tables:
        tr = TableReplacer(t)
        opts: Mapping[str, Any] = {}
        element = replacement_traverse(element, opts, tr.replace)  # type: ignore
    return compiler.visit_select(element, **kw)


@compiles(CreateTable, "mssql")
def compile_mssql_create_table(element: CreateTable, compiler: Any, **kw: Any) -> str:
    """
    Post-process MS-SQL CREATE TABLE DDL:

    1. Strip ON DELETE CASCADE — MS-SQL rejects multiple cascading FK paths to
       the same table (error 1785). Referential integrity is enforced by insert
       order in datafaker, so CASCADE is not needed.
    2. Strip IDENTITY — datafaker generates PK values explicitly via
       ColumnValueProvider.increment(), so auto-generation is not needed and
       would cause INSERT to fail without SET IDENTITY_INSERT ON.
    """
    text: str = compiler.visit_create_table(element, **kw)
    text = text.replace(" ON DELETE CASCADE", "")
    text = re.sub(r"\s+IDENTITY(\(\d+,\s*\d+\))?", "", text)
    return text


@compiles(CreateSchema, "mssql")
def mssql_create_schema(element: CreateSchema, compiler: Any, **kw: Any) -> str:
    """Correct CREATE SCHEMA IF NOT EXISTS."""
    name = element.element.replace("'", "''")
    if element.if_not_exists:
        return f"IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = '{name}') BEGIN EXEC('CREATE SCHEMA {name}') END"
    return f"CREATE SCHEMA {name}"


@compiles(CreateTable, "duckdb")
def remove_on_delete_cascade(element: CreateTable, compiler: Any, **kw: Any) -> str:
    """
    Intercede in compilation for column creation.

    DuckDB does not understand cascades, and we don't care about
    that in datafaker so we remove ``ON DELETE CASCASE``.

    DuckDB does not understand ``SERIAL`` and we don't care
    about autoincrementing, so we will replace it simply with
    ``INTEGER``.

    Ideally ``duckdb_engine`` would remove these for us.
    :param element: The CreateTable being executed.
    :param compiler: Actually a DDLCompiler, but that type is not exported.
    :param kw: Further arguments.
    :return: Corrected SQL.
    """
    text: str = compiler.visit_create_table(element, **kw)
    t2 = serial_re.sub("INTEGER", text)
    return t2.replace(" ON DELETE CASCADE", "")


class StdDev(ColumnElement[float]):  # pylint: disable=too-many-ancestors
    """Represent getting the difference between times in seconds."""

    expr: ColumnElement[float]

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
    ]

    def __init__(
        self,
        expr: ColumnElement[float],
    ):
        """Get a clause for the standard deviation of a sample of values."""
        self.expr = expr

    __sa_operate__ = ColumnElement.operate


@compiles(StdDev)
def compile_stddev(
    element: StdDev, compiler: Any, **kw: Any
) -> str:
    """Create SQL for standard deviation."""
    e = compiler.process(element.expr, **kw)
    return f"STDDEV({e})"


@compiles(StdDev, "mssql")
def compile_stddev_mssql(
    element: StdDev, compiler: Any, **kw: Any
) -> str:
    """MSSQL equivalent: STDEVP."""
    e = compiler.process(element.expr, **kw)
    return f"STDEVP({e})"


class IsNull(ColumnElement[float]):  # pylint: disable=too-many-ancestors
    """Represent IS NULL as an expression."""

    expr: ColumnElement[float]

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
    ]

    def __init__(
        self,
        expr: ColumnElement[float],
    ):
        """Get the clause that is being tested for nullness."""
        self.expr = expr

    __sa_operate__ = ColumnElement.operate


@compiles(IsNull)
def compile_isnull(
    element: IsNull, compiler: Any, **kw: Any
) -> str:
    """Create SQL for IS NULL."""
    e = compiler.process(element.expr, **kw)
    return f"{e} IS NULL"


class IsNotNull(ColumnElement[float]):  # pylint: disable=too-many-ancestors
    """Represent IS NOT NULL as an expression."""

    expr: ColumnElement[float]

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
    ]

    def __init__(
        self,
        expr: ColumnElement[float],
    ):
        """Get the clause that is being tested for nonnullness."""
        self.expr = expr

    __sa_operate__ = ColumnElement.operate


@compiles(IsNotNull)
def compile_isnotnull(
    element: IsNotNull, compiler: Any, **kw: Any
) -> str:
    """Create SQL for IS NULL."""
    e = compiler.process(element.expr, **kw)
    return f"{e} IS NOT NULL"


class Random(ColumnElement[float]):  # pylint: disable=too-many-ancestors
    """Represent a random number between 0 and 1."""

    _traverse_internals = []

    def __init__(
        self,
    ):
        """Get a clause for random values."""


@compiles(Random)
def compile_random(
    element: Random, compiler: Any, **kw: Any
) -> str:
    """Create SQL for random."""
    return "RANDOM()"


@compiles(Random, "mssql")
def compile_random_mssql(
    element: Random, compiler: Any, **kw: Any
) -> str:
    """MSSQL equivalent: RAND."""
    return "RAND()"


class SecondsDifference(ColumnElement[int]):  # pylint: disable=too-many-ancestors
    """Represent getting the difference between times in seconds."""

    expr1: ColumnElement[Date | DateTime]
    expr2: ColumnElement[Date | DateTime]

    _traverse_internals = [
        ("expr1", InternalTraversal.dp_clauseelement),
        ("expr2", InternalTraversal.dp_clauseelement),
    ]

    def __init__(
        self,
        expr1: ColumnElement[Date | DateTime],
        expr2: ColumnElement[Date | DateTime],
    ):
        """
        Get a clause for the number of seconds between two times.

        The interval is from ``expr2`` to ``expr1``.
        """
        self.expr1 = expr1
        self.expr2 = expr2

    __sa_operate__ = ColumnElement.operate


@compiles(SecondsDifference)
def compile_seconds_difference(
    element: SecondsDifference, compiler: Any, **kw: Any
) -> str:
    """Create SQL for the difference between two datetimes in seconds."""
    e1 = compiler.process(element.expr1, **kw)
    e2 = compiler.process(element.expr2, **kw)
    return f"CAST(EXTRACT(EPOCH FROM ({e1})) - EXTRACT(EPOCH FROM ({e2})) AS FLOAT)"


@compiles(SecondsDifference, "mssql")
def compile_seconds_difference_mssql(
    element: SecondsDifference, compiler: Any, **kw: Any
) -> str:
    """MSSQL equivalent: EXTRACT(EPOCH FROM …) is not available; use DATEDIFF."""
    e1 = compiler.process(element.expr1, **kw)
    e2 = compiler.process(element.expr2, **kw)
    return f"CAST(DATEDIFF(second, {e2}, {e1}) AS FLOAT)"
