from enum import StrEnum
import parsy
import re
from sqlalchemy import MetaData, Table, Column, Dialect
from sqlalchemy.dialects import oracle, postgresql
from sqlalchemy.sql import sqltypes
from sqlalchemy.sql.compiler import StrSQLTypeCompiler

type table_component_t = dict[str, any]
type table_t = dict[str, table_component_t]

def simple(type_):
    """
    Parses a simple sqltypes type.
    For example, simple(sqltypes.UUID) takes the string "UUID" and outputs
    a UUID class, or fails with any other string.
    """
    return parsy.string(type_.__name__).result(type_)

def integer():
    """
    Parses an integer, outputting that integer.
    """
    return parsy.regex(r"-?[0-9]+").map(int)

def integer_arguments():
    """
    Parses a list of integers.
    The integers are surrounded by brackets and separated by
    a comma and space.
    """
    return parsy.string("(") >> (
        integer().sep_by(parsy.string(", "))
    ) << parsy.string(")")

def numeric_type(type_):
    """
    Parses TYPE_NAME, TYPE_NAME(2) or TYPE_NAME(2,3)
    passing any arguments to the TYPE_NAME constructor.
    """
    return parsy.string(type_.__name__
    ) >> integer_arguments().optional([]).combine(type_)

def string_type(type_):
    @parsy.generate(type_.__name__)
    def st_parser():
        """
        Parses TYPE_NAME, TYPE_NAME(32), TYPE_NAME COLLATE "fr"
        or TYPE_NAME(32) COLLATE "fr"
        """
        yield parsy.string(type_.__name__)
        length: int | None = yield (
            parsy.string("(") >> integer() << parsy.string(")")
        ).optional()
        collation: str | None = yield (
            parsy.string(' COLLATE "') >> parsy.regex(r'[^"]*') << parsy.string('"')
        ).optional()
        return type_(length=length, collation=collation)
    return st_parser

def time_type(type_, pg_type):
    @parsy.generate(type_.__name__)
    def pgt_parser():
        """
        Parses TYPE_NAME, TYPE_NAME(32), TYPE_NAME WITH TIMEZONE
        or TYPE_NAME(32) WITH TIMEZONE
        """
        yield parsy.string(type_.__name__)
        precision: int | None = yield (
            parsy.string("(") >> integer() << parsy.string(")")
        ).optional()
        timezone: str | None = yield (
            parsy.string(" WITH TIME ZONE").result(True)
        ).optional(False)
        if precision is None and not timezone:
            # normal sql type
            return type_
        return pg_type(precision=precision, timezone=timezone)
    return pgt_parser

SIMPLE_TYPE_PARSER = parsy.alt(
    parsy.string("DOUBLE PRECISION").result(sqltypes.DOUBLE_PRECISION), # must be before DOUBLE
    simple(sqltypes.FLOAT),
    simple(sqltypes.DOUBLE),
    simple(sqltypes.INTEGER),
    simple(sqltypes.SMALLINT),
    simple(sqltypes.BIGINT),
    simple(sqltypes.DATETIME),
    simple(sqltypes.DATE),
    simple(sqltypes.CLOB),
    simple(oracle.NCLOB),
    simple(sqltypes.UUID),
    simple(sqltypes.BLOB),
    simple(sqltypes.BOOLEAN),
    simple(postgresql.TSVECTOR),
    simple(postgresql.BYTEA),
    numeric_type(sqltypes.NUMERIC),
    numeric_type(sqltypes.DECIMAL),
    string_type(sqltypes.CHAR),
    string_type(sqltypes.NCHAR),
    string_type(sqltypes.VARCHAR),
    string_type(sqltypes.NVARCHAR),
    string_type(sqltypes.TEXT),
    time_type(sqltypes.TIMESTAMP, postgresql.types.TIMESTAMP),
    time_type(sqltypes.TIME, postgresql.types.TIME),
)

@parsy.generate
def type_parser():
    base = yield SIMPLE_TYPE_PARSER
    dimensions = yield parsy.string("[]").many().map(len)
    if dimensions == 0:
        return base
    return postgresql.ARRAY(base, dimensions=dimensions)

def column_to_dict(column: Column, dialect: Dialect) -> str:
    type_ = column.type
    if isinstance(type_, postgresql.DOMAIN):
        # Instead of creating a restricted type, we'll just use the base type.
        # It might be better to use the actual type if we could find a good way
        # to compile it and also parse the compiled string.
        type_ = type_.data_type
    if isinstance(type_, postgresql.ENUM):
        compiled = "TEXT"
    else:
        compiled = dialect.type_compiler_instance.process(type_)
    return {
        "type": compiled,
        "primary": column.primary_key,
        "nullable": column.nullable,
        "unique": column.unique
    }

def dict_to_column(name, rep: dict) -> Column:
    type_sql = rep["type"]
    try:
        type_ = type_parser.parse(type_sql)
    except parsy.ParseError as e:
        print(f"Failed to parse {type_sql}")
        raise e
    return Column(
        name=name,
        type_=type_,
        primary_key=rep.get("primary", False),
        nullable=rep.get("nullable", None),
        unique=rep.get("unique", None),
    )

def table_to_dict(table: Table, dialect: Dialect) -> table_t:
    """
    Converts a SQL Alchemy Table object into a
    Python object ready for conversion to YAML.
    """
    return {
        "columns": {
            str(column.key): column_to_dict(column, dialect)
            for (k, column) in table.columns.items()
        },
        "schema": table.schema
    }

def dict_to_table(name: str, meta: MetaData, table_dict: table_t) -> Table:
    return Table(
        name,
        meta,
        *[ dict_to_column(name, col)
            for (name, col) in table_dict.get("columns", {}).items()
        ],
        schema=table_dict.get("schema")
    )

def metadata_to_dict(meta: MetaData, dsn: str, dialect: Dialect) -> dict[str, table_t]:
    """
    Converts a SQL Alchemy MetaData object into
    a Python object ready for conversion to YAML.
    """
    return {
        "tables": {
            str(table.name): table_to_dict(table, dialect)
            for (k, table) in meta.tables.items()
        },
        "dsn": dsn,
        "schema": meta.schema
    }

def dict_to_metadata(obj: dict[str, table_t]) -> MetaData:
    """
    Converts a dict to a SQL Alchemy MetaData object.
    """
    table_dict = obj.get("tables", {})
    schema = obj.get("schema", "public")
    meta = MetaData(schema=schema)
    for (k, td) in table_dict.items():
        dict_to_table(k, meta, td)
    return meta
