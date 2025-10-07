import typing
from typing import Callable, Protocol

import parsy
from sqlalchemy import Column, Dialect, Engine, ForeignKey, MetaData, Table
from sqlalchemy.dialects import oracle, postgresql
from sqlalchemy.sql import schema, sqltypes

from datafaker.utils import make_foreign_key_name

table_t = dict[str, typing.Any]


# We will change this to parsy.Parser when parsy exports its types properly
ParserType = typing.Any


def simple(type_: type) -> ParserType:
    """
    Parses a simple sqltypes type.
    For example, simple(sqltypes.UUID) takes the string "UUID" and outputs
    a UUID class, or fails with any other string.
    """
    return parsy.string(type_.__name__).result(type_)


def integer() -> ParserType:
    """
    Parses an integer, outputting that integer.
    """
    return parsy.regex(r"-?[0-9]+").map(int)


def integer_arguments() -> ParserType:
    """
    Parses a list of integers.
    The integers are surrounded by brackets and separated by
    a comma and space.
    """
    return (
        parsy.string("(") >> (integer().sep_by(parsy.string(", "))) << parsy.string(")")
    )


def numeric_type(type_: type) -> ParserType:
    """
    Parses TYPE_NAME, TYPE_NAME(2) or TYPE_NAME(2,3)
    passing any arguments to the TYPE_NAME constructor.
    """
    return parsy.string(type_.__name__) >> integer_arguments().optional([]).combine(
        type_
    )


def string_type(type_: type) -> ParserType:
    @parsy.generate(type_.__name__)
    def st_parser() -> typing.Generator[ParserType, None, typing.Any]:
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


def time_type(type_: type, pg_type: type) -> ParserType:
    @parsy.generate(type_.__name__)
    def pgt_parser() -> typing.Generator[ParserType, None, typing.Any]:
        """
        Parses TYPE_NAME, TYPE_NAME(32), TYPE_NAME WITH TIME ZONE
        or TYPE_NAME(32) WITH TIME ZONE
        """
        yield parsy.string(type_.__name__)
        precision: int | None = yield (
            parsy.string("(") >> integer() << parsy.string(")")
        ).optional()
        timezone: str | None = yield (
            parsy.string(" WITH")
            >> (parsy.string(" ").result(True) | parsy.string("OUT ").result(False))
            << parsy.string("TIME ZONE")
        ).optional(False)
        if precision is None and not timezone:
            # normal sql type
            return type_
        return pg_type(precision=precision, timezone=timezone)

    return pgt_parser


SIMPLE_TYPE_PARSER = parsy.alt(
    parsy.string("DOUBLE PRECISION").result(
        sqltypes.DOUBLE_PRECISION
    ),  # must be before DOUBLE
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
    simple(postgresql.CIDR),
    numeric_type(sqltypes.NUMERIC),
    numeric_type(sqltypes.DECIMAL),
    numeric_type(postgresql.BIT),
    numeric_type(postgresql.REAL),
    string_type(sqltypes.CHAR),
    string_type(sqltypes.NCHAR),
    string_type(sqltypes.VARCHAR),
    string_type(sqltypes.NVARCHAR),
    string_type(sqltypes.TEXT),
    time_type(sqltypes.TIMESTAMP, postgresql.types.TIMESTAMP),
    time_type(sqltypes.TIME, postgresql.types.TIME),
)


@parsy.generate
def type_parser() -> ParserType:
    base = yield SIMPLE_TYPE_PARSER
    dimensions = yield parsy.string("[]").many().map(len)
    if dimensions == 0:
        return base
    return postgresql.ARRAY(base, dimensions=dimensions)


def column_to_dict(column: Column, dialect: Dialect) -> dict[str, typing.Any]:
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
    result = {
        "type": compiled,
        "primary": column.primary_key,
        "nullable": column.nullable,
    }
    foreign_keys = [str(fk.target_fullname) for fk in column.foreign_keys]
    if foreign_keys:
        result["foreign_keys"] = foreign_keys
    return result


def dict_to_column(
    table_name: str,
    col_name: str,
    rep: dict,
    ignore_fk: Callable[[str], bool],
) -> Column:
    type_sql = rep["type"]
    try:
        type_ = type_parser.parse(type_sql)
    except parsy.ParseError as e:
        print(f"Failed to parse {type_sql}")
        raise e
    if "foreign_keys" in rep:
        args = [
            ForeignKey(
                fk,
                name=make_foreign_key_name(table_name, col_name),
                ondelete="CASCADE",
            )
            for fk in rep["foreign_keys"]
            if not ignore_fk(fk)
        ]
    else:
        args = []
    return Column(
        *args,
        name=col_name,
        type_=type_,
        primary_key=rep.get("primary", False),
        nullable=rep.get("nullable", None),
    )


def dict_to_unique(rep: dict) -> schema.UniqueConstraint:
    return schema.UniqueConstraint(*rep.get("columns", []), name=rep.get("name", None))


def unique_to_dict(constraint: schema.UniqueConstraint) -> dict:
    return {
        "name": constraint.name,
        "columns": [str(col.name) for col in constraint.columns],
    }


def table_to_dict(table: Table, dialect: Dialect) -> table_t:
    """
    Converts a SQL Alchemy Table object into a
    Python object ready for conversion to YAML.
    """
    return {
        "columns": {
            str(column.key): column_to_dict(column, dialect)
            for column in table.columns.values()
        },
        "unique": [
            unique_to_dict(constraint)
            for constraint in table.constraints
            if isinstance(constraint, schema.UniqueConstraint)
        ],
    }


def dict_to_table(
    name: str,
    meta: MetaData,
    table_dict: table_t,
    ignore_fk: Callable[[str], bool],
) -> Table:
    return Table(
        name,
        meta,
        *[
            dict_to_column(name, colname, col, ignore_fk)
            for (colname, col) in table_dict.get("columns", {}).items()
        ],
        *[dict_to_unique(constraint) for constraint in table_dict.get("unique", [])],
    )


def metadata_to_dict(
    meta: MetaData, schema_name: str | None, engine: Engine
) -> dict[str, typing.Any]:
    """
    Converts a SQL Alchemy MetaData object into
    a Python object ready for conversion to YAML.
    """
    return {
        "tables": {
            str(table.name): table_to_dict(table, engine.dialect)
            for table in meta.tables.values()
        },
        "dsn": str(engine.url),
        "schema": schema_name,
    }


def should_ignore_fk(fk: str, tables_dict: dict[str, table_t]) -> bool:
    """
    Tell if this foreign key should be ignored because it points to an
    ignored table.
    """
    fk_bits = fk.split(".", 2)
    if len(fk_bits) != 2:
        return True
    if fk_bits[0] not in tables_dict:
        return False
    return bool(tables_dict[fk_bits[0]].get("ignore", False))


def dict_to_metadata(obj: dict, config_for_output: dict | None = None) -> MetaData:
    """
    Converts a dict to a SQL Alchemy MetaData object.

    :param config_for_output: The configuration object. Should be None if
    the metadata object is being used for connecting to the source database.
    If it is being used for connecting to the destination database this
    configuration will be used to make sure that there is no foreign key
    constraint to an ignored table.
    """
    tables_dict = obj.get("tables", {})
    if config_for_output and "tables" in config_for_output:
        tables_config = config_for_output["tables"]
        ignore_fk = lambda fk: should_ignore_fk(fk, tables_config)
    else:
        ignore_fk = lambda _: False
    meta = MetaData()
    for k, td in tables_dict.items():
        dict_to_table(k, meta, td, ignore_fk)
    return meta
