"""Dialect differences."""

import re
from collections.abc import Mapping
from typing import Any, Optional, TypeVar

import snowflake.sqlalchemy.custom_types as snowflake_types
from sqlalchemy import Column, Select, Table
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
from sqlalchemy.types import Date, DateTime, String

T = TypeVar("T")

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
    Post-process MS-SQL CREATE TABLE DDL.

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
def mssql_create_schema(element: CreateSchema, _compiler: Any, **_kw: Any) -> str:
    """Correct CREATE SCHEMA IF NOT EXISTS."""
    name = element.element.replace("'", "''")
    if element.if_not_exists:
        return (
            "IF NOT EXISTS (SELECT 1 FROM sys.schemas"
            f" WHERE name = '{name}')"
            f" BEGIN EXEC('CREATE SCHEMA {name}') END"
        )
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


class StdDev(ColumnElement[float]):  # pylint: disable=too-many-ancestors
    """Represent getting the difference between times in seconds."""

    expr: ColumnElement[int | float] | SecondsDifference

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
    ]

    def __init__(
        self,
        expr: ColumnElement[int | float] | SecondsDifference,
    ):
        """Get a clause for the standard deviation of a sample of values."""
        self.expr = expr

    __sa_operate__ = ColumnElement.operate


@compiles(StdDev)
def compile_stddev(element: StdDev, compiler: Any, **kw: Any) -> str:
    """Create SQL for standard deviation."""
    e = compiler.process(element.expr, **kw)
    return f"STDDEV({e})"


@compiles(StdDev, "mssql")
def compile_stddev_mssql(element: StdDev, compiler: Any, **kw: Any) -> str:
    """MSSQL equivalent: STDEV."""
    e = compiler.process(element.expr, **kw)
    return f"STDEV({e})"


class WordCount(ColumnElement[int]):  # pylint: disable=too-many-ancestors
    """Count whitespace-separated words in a text expression."""

    expr: ColumnElement[str]

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
    ]

    def __init__(self, expr: ColumnElement[str]):
        """Get a clause for the number of whitespace-separated words in ``expr``."""
        self.expr = expr

    __sa_operate__ = ColumnElement.operate


@compiles(WordCount)
def compile_word_count(element: WordCount, compiler: Any, **kw: Any) -> str:
    """Exact word count via regexp_split_to_array (Postgres/DuckDB)."""
    e = compiler.process(element.expr, **kw)
    return f"array_length(regexp_split_to_array(trim({e}), '\\s+'), 1)"


@compiles(WordCount, "mssql")
def compile_word_count_mssql(element: WordCount, compiler: Any, **kw: Any) -> str:
    """MSSQL equivalent: approximate word count via space-counting.

    No regex/array-split support, so approximate word count as (space
    count in the trimmed text) + 1. This overcounts by one per run of 2+
    consecutive spaces relative to the exact regex-based count used on
    Postgres/DuckDB, but there is no T-SQL primitive for a real
    whitespace-run split.
    """
    e = compiler.process(element.expr, **kw)
    trimmed = f"TRIM({e})"
    return (
        f"(CASE WHEN LEN({trimmed}) = 0 THEN 0 ELSE "
        f"LEN({trimmed}) - LEN(REPLACE({trimmed}, ' ', '')) + 1 END)"
    )


class SentenceCount(ColumnElement[int]):  # pylint: disable=too-many-ancestors
    """Count '.'/'!'/'?'-delimited sentences in a text expression."""

    expr: ColumnElement[str]

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
    ]

    def __init__(self, expr: ColumnElement[str]):
        """Get a clause for the number of sentences in ``expr``."""
        self.expr = expr

    __sa_operate__ = ColumnElement.operate


@compiles(SentenceCount)
def compile_sentence_count(element: SentenceCount, compiler: Any, **kw: Any) -> str:
    """Exact sentence count via regexp_split_to_array (Postgres/DuckDB)."""
    e = compiler.process(element.expr, **kw)
    return (
        f"array_length(regexp_split_to_array(trim({e}, ' .!?\\t\\n\\r'), '[.!?]+'), 1)"
    )


@compiles(SentenceCount, "mssql")
def compile_sentence_count_mssql(
    element: SentenceCount, compiler: Any, **kw: Any
) -> str:
    """MSSQL equivalent: approximate sentence count via delimiter-counting.

    No regex support, so approximate sentence count by normalizing '!'/'?'
    to '.' and counting '.' characters, after trimming whitespace and
    sentence-enders from both ends (so a leading/trailing sentence-ender
    doesn't add a spurious segment, matching the Postgres/DuckDB
    behaviour). This overcounts by one per run of 2+ consecutive delimiter
    characters (e.g. "Wow!!") relative to the exact regex-based count, but
    there is no T-SQL primitive for regex-run splitting.
    """
    e = compiler.process(element.expr, **kw)
    trim_chars = "' .!?' + CHAR(9) + CHAR(10) + CHAR(13)"
    trimmed = f"TRIM({trim_chars} FROM {e})"
    normalized = f"REPLACE(REPLACE({trimmed}, '!', '.'), '?', '.')"
    return (
        f"(CASE WHEN LEN({trimmed}) = 0 THEN 0 ELSE "
        f"LEN({normalized}) - LEN(REPLACE({normalized}, '.', '')) + 1 END)"
    )


class StripWhitespace(ColumnElement[str]):  # pylint: disable=too-many-ancestors
    """Strip leading/trailing whitespace from a text expression.

    Approximates Python's ``str.strip()`` by removing spaces, tabs, line
    feeds and carriage returns from both ends. (``str.strip()`` also removes
    some rarer whitespace, such as form feeds and Unicode spaces.)
    """

    expr: ColumnElement[str]
    type = String()

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
    ]

    def __init__(self, expr: ColumnElement[str]):
        """Get a clause for ``expr`` with surrounding whitespace removed."""
        self.expr = expr

    __sa_operate__ = ColumnElement.operate


@compiles(StripWhitespace)
def compile_strip_whitespace(element: StripWhitespace, compiler: Any, **kw: Any) -> str:
    """Postgres/DuckDB/Snowflake: two-argument TRIM.

    The whitespace characters are built with CHR() rather than written as
    backslash escapes, because a backslash has no special meaning in a
    standard SQL string literal (so a backslash-t would trim backslashes
    and the letter t, not tabs).
    """
    e = compiler.process(element.expr, **kw)
    return f"TRIM({e}, ' ' || CHR(9) || CHR(10) || CHR(13))"


@compiles(StripWhitespace, "mssql")
def compile_strip_whitespace_mssql(
    element: StripWhitespace, compiler: Any, **kw: Any
) -> str:
    """MSSQL equivalent: TRIM(<characters> FROM <expr>)."""
    e = compiler.process(element.expr, **kw)
    return f"TRIM(' ' + CHAR(9) + CHAR(10) + CHAR(13) FROM {e})"


class AfterCharacter(ColumnElement[str]):  # pylint: disable=too-many-ancestors
    """The text after the first (or last) occurrence of a character.

    Evaluates to NULL when the character does not occur in the text or when
    nothing follows it, matching the "yield nothing" behaviour of
    ``str.partition``/``str.rsplit``-based feature extractors.
    """

    expr: ColumnElement[str]
    char: str
    last: bool
    type = String()

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
        ("char", InternalTraversal.dp_string),
        ("last", InternalTraversal.dp_boolean),
    ]

    def __init__(self, expr: ColumnElement[str], char: str, last: bool = False):
        """
        Get a clause for the part of ``expr`` after ``char``.

        :param expr: The text expression to search.
        :param char: The single character to search for.
        :param last: Split at the last occurrence of ``char`` rather than the
            first.
        """
        if len(char) != 1:
            raise ValueError(f"AfterCharacter needs a single character, not {char!r}")
        self.expr = expr
        self.char = char
        self.last = last

    __sa_operate__ = ColumnElement.operate


@compiles(AfterCharacter)
def compile_after_character(element: AfterCharacter, compiler: Any, **kw: Any) -> str:
    """Postgres/DuckDB/Snowflake: POSITION/SUBSTR, or REVERSE/SPLIT_PART for last."""
    e = compiler.process(element.expr, **kw)
    c = compiler.render_literal_value(element.char, String())
    if element.last:
        after = f"REVERSE(SPLIT_PART(REVERSE({e}), {c}, 1))"
    else:
        after = f"SUBSTR({e}, POSITION({c} IN {e}) + 1)"
    return f"(CASE WHEN POSITION({c} IN {e}) > 0 THEN NULLIF({after}, '') END)"


@compiles(AfterCharacter, "mssql")
def compile_after_character_mssql(
    element: AfterCharacter, compiler: Any, **kw: Any
) -> str:
    """MSSQL equivalent: CHARINDEX/SUBSTRING, or REVERSE/RIGHT for last.

    DATALENGTH (bytes) rather than LEN is used as SUBSTRING's length because
    LEN ignores trailing spaces; it is always at least the number of
    characters left. Note that MSSQL treats ``''`` as equal to an
    all-space string, so an all-space remainder also becomes NULL - this
    cannot happen when ``expr`` has been through ``StripWhitespace``.
    """
    e = compiler.process(element.expr, **kw)
    c = compiler.render_literal_value(element.char, String())
    if element.last:
        after = f"RIGHT({e}, CHARINDEX({c}, REVERSE({e})) - 1)"
    else:
        after = f"SUBSTRING({e}, CHARINDEX({c}, {e}) + 1, DATALENGTH({e}))"
    return f"(CASE WHEN CHARINDEX({c}, {e}) > 0 THEN NULLIF({after}, '') END)"


class BeforeCharacter(ColumnElement[str]):  # pylint: disable=too-many-ancestors
    """The text before the first occurrence of a character.

    Evaluates to NULL when the character does not occur in the text or when
    nothing precedes it, matching the "yield nothing" behaviour of
    ``str.partition``-based feature extractors.
    """

    expr: ColumnElement[str]
    char: str
    type = String()

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
        ("char", InternalTraversal.dp_string),
    ]

    def __init__(self, expr: ColumnElement[str], char: str):
        """
        Get a clause for the part of ``expr`` before the first ``char``.

        :param expr: The text expression to search.
        :param char: The single character to search for.
        """
        if len(char) != 1:
            raise ValueError(f"BeforeCharacter needs a single character, not {char!r}")
        self.expr = expr
        self.char = char

    __sa_operate__ = ColumnElement.operate


@compiles(BeforeCharacter)
def compile_before_character(element: BeforeCharacter, compiler: Any, **kw: Any) -> str:
    """Postgres/DuckDB/Snowflake: POSITION/SUBSTR."""
    e = compiler.process(element.expr, **kw)
    c = compiler.render_literal_value(element.char, String())
    before = f"SUBSTR({e}, 1, POSITION({c} IN {e}) - 1)"
    return f"(CASE WHEN POSITION({c} IN {e}) > 0 THEN NULLIF({before}, '') END)"


@compiles(BeforeCharacter, "mssql")
def compile_before_character_mssql(
    element: BeforeCharacter, compiler: Any, **kw: Any
) -> str:
    """MSSQL equivalent: CHARINDEX/LEFT.

    As with ``AfterCharacter``, MSSQL treats ``''`` as equal to an all-space
    string, so an all-space prefix also becomes NULL - this cannot happen
    when ``expr`` has been through ``StripWhitespace``.
    """
    e = compiler.process(element.expr, **kw)
    c = compiler.render_literal_value(element.char, String())
    before = f"LEFT({e}, CHARINDEX({c}, {e}) - 1)"
    return f"(CASE WHEN CHARINDEX({c}, {e}) > 0 THEN NULLIF({before}, '') END)"


class IsNull(ColumnElement[bool]):  # pylint: disable=too-many-ancestors
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
def compile_isnull(element: IsNull, compiler: Any, **kw: Any) -> str:
    """Create SQL for IS NULL."""
    e = compiler.process(element.expr, **kw)
    return f"{e} IS NULL"


class IsNotNull(ColumnElement[bool]):  # pylint: disable=too-many-ancestors
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
def compile_isnotnull(element: IsNotNull, compiler: Any, **kw: Any) -> str:
    """Create SQL for IS NULL."""
    e = compiler.process(element.expr, **kw)
    return f"{e} IS NOT NULL"


class IsNumeric(ColumnElement[bool]):  # pylint: disable=too-many-ancestors
    """Tell if a column is not Null, NaN or Infinity."""

    expr: ColumnElement[float]

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
    ]

    def __init__(
        self,
        expr: ColumnElement[float],
    ):
        """Get the clause that is being tested."""
        self.expr = expr

    __sa_operate__ = ColumnElement.operate


@compiles(IsNumeric)
def compile_isnumeric(element: IsNumeric, compiler: Any, **kw: Any) -> str:
    """Create SQL for IsNumeric."""
    e = compiler.process(element.expr, **kw)
    return f"COALESCE({e} != {e} + 1, False)"


@compiles(IsNumeric, "mssql")
def compile_isnumeric_mssql(element: IsNumeric, compiler: Any, **kw: Any) -> str:
    """Create SQL for IsNumeric for MSSQL."""
    e = compiler.process(element.expr, **kw)
    return f"ISNULL({e} * 0, 1) != 1"


class IsPositive(ColumnElement[bool]):  # pylint: disable=too-many-ancestors
    """Tell if a column is not Null, NaN, Infinity, negative or zero."""

    expr: ColumnElement[float]

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
    ]

    def __init__(
        self,
        expr: ColumnElement[float],
    ):
        """Get the clause that is being tested."""
        self.expr = expr

    __sa_operate__ = ColumnElement.operate


@compiles(IsPositive)
def compile_ispositive(element: IsPositive, compiler: Any, **kw: Any) -> str:
    """Create SQL for IsPositive."""
    e = compiler.process(element.expr, **kw)
    return f"COALESCE({e} != {e} + 1 AND {e} > 0, False)"


@compiles(IsPositive, "mssql")
def compile_ispositive_mssql(element: IsPositive, compiler: Any, **kw: Any) -> str:
    """Create SQL for IsPositive for MSSQL."""
    e = compiler.process(element.expr, **kw)
    return f"ISNULL({e}, -1) > 0"


class LogNatural(ColumnElement[float]):  # pylint: disable=too-many-ancestors
    """Calculate the logarithm base e."""

    expr: ColumnElement[float]

    _traverse_internals = [
        ("expr", InternalTraversal.dp_clauseelement),
    ]

    def __init__(
        self,
        expr: ColumnElement[float],
    ):
        """Get the clause that is being tested."""
        self.expr = expr

    __sa_operate__ = ColumnElement.operate


@compiles(LogNatural)
def compile_lognatural(element: LogNatural, compiler: Any, **kw: Any) -> str:
    """Create SQL for LogNatural."""
    e = compiler.process(element.expr, **kw)
    return f"LN({e})"


@compiles(LogNatural, "mssql")
def compile_lognatural_mssql(element: LogNatural, compiler: Any, **kw: Any) -> str:
    """Create SQL for LogNatural for MSSQL."""
    e = compiler.process(element.expr, **kw)
    return f"LOG({e})"


class Random(ColumnElement[float]):  # pylint: disable=too-many-ancestors
    """Represent a random value suitable for choosing random rows."""

    _traverse_internals = []

    def __init__(
        self,
    ):
        """Get a clause for random values."""

    __sa_operate__ = ColumnElement.operate


class NullIf(ColumnElement[Optional[T]]):  # pylint: disable=too-many-ancestors
    """Represent NULLIF."""

    expr1: ColumnElement[T]
    expr2: ColumnElement[T]

    _traverse_internals = [
        ("expr1", InternalTraversal.dp_clauseelement),
        ("expr2", InternalTraversal.dp_clauseelement),
    ]

    def __init__(
        self,
        expr1: ColumnElement[T],
        expr2: ColumnElement[T],
    ):
        """
        Get a NULLIF clause.

        If ``expr1`` = ``expr2`` the result is NULL, otherwise ``expr1``.
        """
        self.expr1 = expr1
        self.expr2 = expr2

    __sa_operate__ = ColumnElement.operate


@compiles(NullIf)
def compile_null_if(element: NullIf, compiler: Any, **kw: Any) -> str:
    """Create SQL for NULLIF."""
    e1 = compiler.process(element.expr1, **kw)
    e2 = compiler.process(element.expr2, **kw)
    return f"NULLIF({e1}, {e2})"


@compiles(Random)
def compile_random(_element: Random, _compiler: Any, **_kw: Any) -> str:
    """Create SQL for random."""
    return "RANDOM()"


@compiles(Random, "mssql")
def compile_random_mssql(_element: Random, _compiler: Any, **_kw: Any) -> str:
    """
    MSSQL uses NEWID.

    RAND() is the obvious equivalent, but it does not work because the
    same random number gets chosen for each row.
    """
    return "NEWID()"


def random_sample(query: Select, n: int | ColumnElement[Any]) -> Select:
    """Return a random sample of up to n rows of the query.

    The rows are shuffled before limiting, so the sample isn't biased
    towards the order in which the database happens to store them.
    """
    return query.order_by(Random()).limit(n)


@compiles(snowflake_types.TIMESTAMP_LTZ)
def compile_timestampltz_generic(
    _element: snowflake_types.TIMESTAMP_LTZ, compiler: Any, **kw: Any
) -> str:
    """Render TIMESTAMP_LTZ as DateTime."""
    return compiler.process(DateTime(timezone=False), **kw)


@compiles(snowflake_types.TIMESTAMP_LTZ, "snowflake")
def compile_timestampltz_ddb(
    element: snowflake_types.TIMESTAMP_LTZ, compiler: Any, **kw: Any
) -> str:
    """Render TIMESTAMP_LTZ as TIMESTAMP_LTZ."""
    return compiler.visit_TIMESTAMP_LTZ(element, **kw)
