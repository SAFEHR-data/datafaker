"""Functions to make a module of generator classes."""
import asyncio
import decimal
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Final, Mapping, Optional, Sequence, Tuple, Type

import pandas as pd
import snsql
import yaml
from black import FileMode, format_str
from jinja2 import Environment, FileSystemLoader, Template
from mimesis.providers.base import BaseProvider
from sqlalchemy import CursorResult, Engine, MetaData, UniqueConstraint, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.schema import Column, Table
from sqlalchemy.sql import Executable, sqltypes
from typing_extensions import Self

from datafaker import providers
from datafaker.settings import get_settings
from datafaker.utils import (
    MaybeAsyncEngine,
    create_db_engine,
    download_table,
    get_columns_assigned,
    get_property,
    get_related_table_names,
    get_row_generators,
    get_sync_engine,
    get_vocabulary_table_names,
    logger,
    make_primary_key_name,
    split_foreign_key_target,
)

from .serialize_metadata import metadata_to_dict

PROVIDER_IMPORTS: Final[list[str]] = []
for entry_name, entry in inspect.getmembers(providers, inspect.isclass):
    if issubclass(entry, BaseProvider) and entry.__module__ == "datafaker.providers":
        PROVIDER_IMPORTS.append(entry_name)

TEMPLATE_DIRECTORY: Final[Path] = Path(__file__).parent / "templates/"
DF_TEMPLATE_FILENAME: Final[str] = "df.py.j2"


@dataclass
class VocabularyTableGeneratorInfo:
    """Contains the df.py content related to vocabulary tables."""

    variable_name: str
    table_name: str
    dictionary_entry: str


@dataclass
class FunctionCall:
    """Contains the df.py content related function calls."""

    function_name: str
    argument_values: list[str]


@dataclass
class RowGeneratorInfo:
    """Contains the df.py content related to row generators of a table."""

    variable_names: list[str]
    function_call: FunctionCall
    primary_key: bool = False


@dataclass
class ColumnChoice:
    """Choose columns based on a random number in [0,1)."""

    function_name: str
    argument_values: list[str]


def make_column_choices(
    table_config: Mapping[str, Any],
) -> list[ColumnChoice]:
    """
    Convert ``missingness_generators`` from ``config.yaml`` into functions to call.

    :param table_config: The ``tables`` part of ``config.yaml``.
    :return: A list of ``ColumnChoice`` objects; that is, descriptions of
    functions and their arguments to call to reveal a list of columns that
    should have values generated for them.
    """
    return [
        ColumnChoice(
            function_name=mg["name"],
            argument_values=[f"{k}={v}" for k, v in mg.get("kwargs", {}).items()],
        )
        for mg in table_config.get("missingness_generators", [])
        if "name" in mg
    ]


@dataclass
class _PrimaryConstraint:
    """
    Describes a Uniqueness constraint for a multi-column primary key.

    Not a real constraint, but enough to write df.py.
    """

    columns: list[Column]
    name: str


@dataclass
class TableGeneratorInfo:
    """Contains the df.py content related to regular tables."""

    class_name: str
    table_name: str
    nonnull_columns: set[str]
    column_choices: list[ColumnChoice]
    rows_per_pass: int
    row_gens: list[RowGeneratorInfo] = field(default_factory=list)
    unique_constraints: Sequence[UniqueConstraint | _PrimaryConstraint] = field(
        default_factory=list
    )


@dataclass
class StoryGeneratorInfo:
    """Contains the df.py content related to story generators."""

    wrapper_name: str
    function_call: FunctionCall
    num_stories_per_pass: int


def _render_value(v: Any) -> str:
    if isinstance(v, list):
        return "[" + ", ".join(_render_value(x) for x in v) + "]"
    if isinstance(v, set):
        return "{" + ", ".join(_render_value(x) for x in v) + "}"
    if isinstance(v, dict):
        return (
            "{" + ", ".join(f"{repr(k)}:{_render_value(x)}" for k, x in v.items()) + "}"
        )
    if isinstance(v, str):
        return v
    return str(v)


def _get_function_call(
    function_name: str,
    positional_arguments: Optional[Sequence[Any]] = None,
    keyword_arguments: Optional[Mapping[str, Any]] = None,
) -> FunctionCall:
    if positional_arguments is None:
        positional_arguments = []

    if keyword_arguments is None:
        keyword_arguments = {}

    argument_values: list[str] = [str(value) for value in positional_arguments]
    argument_values += [
        f"{key}={_render_value(value)}" for key, value in keyword_arguments.items()
    ]

    return FunctionCall(function_name=function_name, argument_values=argument_values)


def _get_row_generator(
    table_config: Mapping[str, Any],
) -> tuple[list[RowGeneratorInfo], list[str]]:
    """Get the row generators information, for the given table."""
    row_gen_info: list[RowGeneratorInfo] = []
    columns_covered = []
    for name, gen_conf in get_row_generators(table_config):
        columns_assigned = list(get_columns_assigned(gen_conf))
        keyword_arguments: Mapping[str, Any] = gen_conf.get("kwargs", {})
        positional_arguments: Sequence[str] = gen_conf.get("args", [])
        columns_covered += columns_assigned
        row_gen_info.append(
            RowGeneratorInfo(
                variable_names=columns_assigned,
                function_call=_get_function_call(
                    name, positional_arguments, keyword_arguments
                ),
            )
        )
    return row_gen_info, columns_covered


def _get_default_generator(column: Column) -> RowGeneratorInfo:
    """Get default generator information, for the given column."""
    # If it's a primary key column, we presume that primary keys are populated
    # automatically.

    # If it's a foreign key column, pull random values from the column it
    # references.
    variable_names: list[str] = []
    generator_function: str = ""
    generator_arguments: list[str] = []

    if column.foreign_keys:
        if len(column.foreign_keys) > 1:
            raise NotImplementedError(
                "Can't handle multiple foreign keys for one column."
            )
        fkey = next(iter(column.foreign_keys))
        (target_table_name, target_column_name) = split_foreign_key_target(fkey.target_fullname)

        variable_names = [column.name]
        generator_function = "generic.column_value_provider.column_value"
        generator_arguments = [
            "dst_db_conn",
            f"metadata.tables['{target_table_name}']",
            f'"{target_column_name}"',
        ]
        return RowGeneratorInfo(
            primary_key=column.primary_key,
            variable_names=variable_names,
            function_call=_get_function_call(
                function_name=generator_function,
                positional_arguments=generator_arguments,
            ),
        )

    # Otherwise generate values based on just the datatype of the column.
    (
        variable_names,
        generator_function,
        generator_kwargs,
    ) = _get_provider_for_column(column)

    return RowGeneratorInfo(
        primary_key=column.primary_key,
        variable_names=variable_names,
        function_call=_get_function_call(
            function_name=generator_function, keyword_arguments=generator_kwargs
        ),
    )


def _numeric_generator(column: Column) -> tuple[str, dict[str, str]]:
    """
    Get the default generator name and arguments.

    :param column: The column to get the generator for.
    :return: The name of a generator and its arguments.
    """
    column_type = column.type
    scale = getattr(column_type, "scale", None)
    if scale is None:
        return ("generic.numeric.float_number", {})
    return (
        "generic.numeric.float_number",
        {
            "start": "0",
            "end": str(10**scale - 1),
        },
    )


def _string_generator(column: Column) -> tuple[str, dict[str, str]]:
    """
    Get the name of the default string generator for a column.

    :param column: The column to get the generator for.
    :return: The name of the generator and its arguments.
    """
    column_size: Optional[int] = getattr(column.type, "length", None)
    if column_size is None:
        return ("generic.text.color", {})
    return ("generic.person.password", {"length": str(column_size)})


def _integer_generator(column: Column) -> tuple[str, dict[str, str]]:
    """
    Get the name of the default integer generator.

    :param column: The column to get the generator for.
    :return: A pair consisting of the name of a generator and its
    arguments.
    """
    if not column.primary_key:
        return ("generic.numeric.integer_number", {})
    return (
        "generic.column_value_provider.increment",
        {
            "db_connection": "dst_db_conn",
            "column": f'metadata.tables["{column.table.name}"].columns["{column.name}"]',
        },
    )


_YEAR_SUMMARY_QUERY = (
    "SELECT MIN(y) AS start, MAX(y) AS end FROM "
    "(SELECT EXTRACT(YEAR FROM {column}) AS y FROM {table}) AS years"
)


@dataclass
class GeneratorInfo:
    """Description of a generator."""

    # Name or function to generate random objects of this type (not using summary data)
    generator: str | Callable[[Column], tuple[str, dict[str, str]]]
    # SQL query that gets the data to supply as arguments to the generator
    # ({column} and {table} will be interpolated)
    summary_query: str | None = None
    # Dictionary of the names returned from the summary_query to arg types.
    # An arg type is a callable turning the returned value into a Python type to
    # pass as an argument to the generator.
    arg_types: dict[str, Callable] = field(default_factory=dict)
    # True if we should see if we can treat this column as a choice from a finite set
    numeric: bool = False
    # True if we should see if we can treat this column as an amount with a distribution
    choice: bool = False


def get_result_mappings(
    info: GeneratorInfo, results: CursorResult
) -> dict[str, Any] | None:
    """
    Get a mapping from the results of a database query.

    :return: A Python dictionary converted according to the GeneratorInfo provided.
    """
    kw: dict[str, Any] = {}
    mapping = results.mappings().first()
    if mapping is None:
        return kw
    for k, v in mapping.items():
        if v is None:
            return None
        conv_fn = info.arg_types.get(k, float)
        kw[k] = conv_fn(v)
    return kw


_COLUMN_TYPE_TO_GENERATOR_INFO = {
    sqltypes.Boolean: GeneratorInfo(
        generator="generic.development.boolean",
        choice=True,
    ),
    sqltypes.Date: GeneratorInfo(
        generator="generic.datetime.date",
        summary_query=_YEAR_SUMMARY_QUERY,
        arg_types={"start": int, "end": int},
    ),
    sqltypes.DateTime: GeneratorInfo(
        generator="generic.datetime.datetime",
        summary_query=_YEAR_SUMMARY_QUERY,
        arg_types={"start": int, "end": int},
    ),
    sqltypes.Integer: GeneratorInfo(  # must be before Numeric
        generator=_integer_generator,
        numeric=True,
        choice=True,
    ),
    sqltypes.Numeric: GeneratorInfo(
        generator=_numeric_generator,
        numeric=True,
        choice=True,
    ),
    sqltypes.LargeBinary: GeneratorInfo(
        generator="generic.bytes_provider.bytes",
    ),
    sqltypes.Uuid: GeneratorInfo(
        generator="generic.cryptographic.uuid",
    ),
    postgresql.UUID: GeneratorInfo(
        generator="generic.cryptographic.uuid",
    ),
    sqltypes.String: GeneratorInfo(
        generator=_string_generator,
        choice=True,
    ),
}


def _get_info_for_column_type(column_t: type) -> GeneratorInfo | None:
    """
    Get a generator from a column type.

    Returns either a string representing the callable, or a callable that,
    given the column.type will return a tuple (string representing generator
    callable, dict of keyword arguments to pass to the callable).
    """
    if column_t in _COLUMN_TYPE_TO_GENERATOR_INFO:
        return _COLUMN_TYPE_TO_GENERATOR_INFO[column_t]

    # Search exhaustively for a superclass to the columns actual type
    for key, value in _COLUMN_TYPE_TO_GENERATOR_INFO.items():
        if issubclass(column_t, key):
            return value

    return None


def _get_generator_for_column(
    column_t: type,
) -> str | Callable[[Column], tuple[str, dict[str, str]]] | None:
    """
    Get a generator from a column type.

    Returns either a string representing the callable, or a callable that,
    given the column.type will return a tuple (string representing generator
    callable, dict of keyword arguments to pass to the callable).
    """
    info = _get_info_for_column_type(column_t)
    return None if info is None else info.generator


def _get_generator_and_arguments(column: Column) -> tuple[str | None, dict[str, str]]:
    """
    Get the generator and its arguments from the column type.

    :return: A tuple of a string representing the generator callable and a dict of
    keyword arguments to supply to it.
    """
    generator_function = _get_generator_for_column(type(column.type))

    generator_arguments: dict[str, str] = {}
    if callable(generator_function):
        (generator_function, generator_arguments) = generator_function(column)
    return generator_function, generator_arguments


def _get_provider_for_column(column: Column) -> Tuple[list[str], str, dict[str, str]]:
    """
    Get a default Mimesis provider and its arguments for a SQL column type.

    Args:
        column: SQLAlchemy column object

    Returns:
        Tuple[str, str, list[str]]: Tuple containing the variable names to assign to,
        generator function and any generator arguments.
    """
    variable_names: list[str] = [column.name]

    generator_function, generator_arguments = _get_generator_and_arguments(column)

    # If we still don't have a generator, use null and warn.
    if not generator_function:
        generator_function = "generic.null_provider.null"
        logger.warning(
            "Unsupported SQLAlchemy type %s for column %s. "
            "Setting this column to NULL always, "
            "you may want to configure a row generator for it instead.",
            column.type,
            column.name,
        )

    return variable_names, generator_function, generator_arguments


def _constraint_sort_key(constraint: UniqueConstraint) -> str:
    """Extract a string out of a UniqueConstraint that is unique to that constraint.

    We sort the constraints so that the output of make_tables is deterministic, this is
    the sort key.
    """
    return (
        constraint.name
        if isinstance(constraint.name, str)
        else "_".join(map(str, constraint.columns))
    )


def _get_generator_for_table(
    table_config: Mapping[str, Any],
    table: Table,
) -> TableGeneratorInfo:
    """Get generator information for the given table."""
    unique_constraints = sorted(
        (
            constraint
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        ),
        key=_constraint_sort_key,
    )
    primary_keys = [c for c in table.columns if c.primary_key]
    constraints: Sequence[UniqueConstraint | _PrimaryConstraint] = unique_constraints
    if 1 < len(primary_keys):
        primary_constraint = _PrimaryConstraint(
            columns=primary_keys, name=make_primary_key_name(table.name)
        )
        constraints = unique_constraints + [primary_constraint]
    column_choices = make_column_choices(table_config)
    if column_choices:
        nonnull_columns = {
            str(col.name)
            for col in table.columns
            if not table.columns[col.name].nullable
        }
    else:
        nonnull_columns = {str(col.name) for col in table.columns}
    table_data: TableGeneratorInfo = TableGeneratorInfo(
        table_name=table.name,
        class_name=table.name.title().replace(".", "") + "Generator",
        nonnull_columns=nonnull_columns,
        column_choices=column_choices,
        rows_per_pass=get_property(table_config, "num_rows_per_pass", int, 1),
        unique_constraints=constraints,
    )

    row_gen_info_data, columns_covered = _get_row_generator(table_config)
    table_data.row_gens.extend(row_gen_info_data)

    for column in table.columns:
        if column.name not in columns_covered:
            table_data.row_gens.append(_get_default_generator(column))

    return table_data


def _get_story_generators(config: Mapping) -> list[StoryGeneratorInfo]:
    """Get story generators."""
    generators = []
    for gen in config.get("story_generators", []):
        wrapper_name = "run_" + gen["name"].replace(".", "_").lower()
        generators.append(
            StoryGeneratorInfo(
                wrapper_name=wrapper_name,
                function_call=_get_function_call(
                    function_name=gen["name"],
                    keyword_arguments=gen.get("kwargs"),
                    positional_arguments=gen.get("args"),
                ),
                num_stories_per_pass=gen["num_stories_per_pass"],
            )
        )
    return generators


def make_vocabulary_tables(
    metadata: MetaData,
    config: Mapping,
    overwrite_files: bool,
    compress: bool,
    table_names: set[str] | None = None,
) -> None:
    """Extract the data from the source database for each vocabulary table."""
    settings = get_settings()
    src_dsn: str = settings.src_dsn or ""
    assert src_dsn != "", "Missing SRC_DSN setting."

    engine = get_sync_engine(create_db_engine(src_dsn, schema_name=settings.src_schema))
    vocab_names = get_vocabulary_table_names(config)
    if table_names is None:
        table_names = vocab_names
    else:
        invalid_names = table_names - vocab_names
        if invalid_names:
            logger.error(
                "The following names are not the names of vocabulary tables: %s",
                invalid_names,
            )
            logger.info("Valid names are: %s", vocab_names)
            return
    for table_name in table_names:
        _generate_vocabulary_table(
            metadata.tables[table_name],
            engine,
            overwrite_files=overwrite_files,
            compress=compress,
        )


def make_table_generators(  # pylint: disable=too-many-locals
    metadata: MetaData,
    config: Mapping,
    orm_filename: str,
    config_filename: str,
    src_stats_filename: Optional[str],
) -> str:
    """
    Create datafaker generator classes.

    The orm and vocabulary YAML files must already have been
    generated (by make-tables and make-vocab).

    Args:
      metadata: database ORM
      config: Configuration to control the generator creation.
      orm_filename: "orm.yaml" file path so that the generator
      file can load the MetaData object
      config_filename: "config.yaml" file path so that the generator
      file can load the MetaData object
      src_stats_filename: A filename for where to read src stats from.
        Optional, if `None` this feature will be skipped
      overwrite_files: Whether to overwrite pre-existing vocabulary files

    Returns:
      A string that is a valid Python module, once written to file.
    """
    row_generator_module_name: str = config.get("row_generators_module", None)
    story_generator_module_name = config.get("story_generators_module", None)
    object_instantiation: dict[str, dict] = config.get("object_instantiation", {})
    tables_config = config.get("tables", {})

    tables: list[TableGeneratorInfo] = []
    vocabulary_tables: list[VocabularyTableGeneratorInfo] = []
    vocab_names = get_vocabulary_table_names(config)
    for table_name, table in metadata.tables.items():
        if table_name in vocab_names:
            related = get_related_table_names(table)
            related_non_vocab = related.difference(vocab_names)
            if related_non_vocab:
                logger.warning(
                    "Making table '%s' a vocabulary table requires that also the"
                    " related tables (%s) be also vocabulary tables.",
                    table.name,
                    related_non_vocab,
                )
            vocabulary_tables.append(
                _get_generator_for_existing_vocabulary_table(table)
            )
        else:
            tables.append(
                _get_generator_for_table(
                    tables_config.get(table.name, {}),
                    table,
                )
            )

    story_generators = _get_story_generators(config)

    max_unique_constraint_tries = config.get("max-unique-constraint-tries", None)
    return generate_df_content(
        {
            "provider_imports": PROVIDER_IMPORTS,
            "orm_file_name": orm_filename,
            "config_file_name": config_filename,
            "row_generator_module_name": row_generator_module_name,
            "story_generator_module_name": story_generator_module_name,
            "object_instantiation": object_instantiation,
            "src_stats_filename": src_stats_filename,
            "tables": tables,
            "vocabulary_tables": vocabulary_tables,
            "story_generators": story_generators,
            "max_unique_constraint_tries": max_unique_constraint_tries,
        }
    )


def generate_df_content(template_context: Mapping[str, Any]) -> str:
    """Generate the content of the df.py file as a string."""
    environment: Environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIRECTORY),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    df_template: Template = environment.get_template(DF_TEMPLATE_FILENAME)
    template_output: str = df_template.render(template_context)
    return format_str(template_output, mode=FileMode())


def _get_generator_for_existing_vocabulary_table(
    table: Table,
) -> VocabularyTableGeneratorInfo:
    """Turn an existing vocabulary YAML file into a VocabularyTableGeneratorInfo."""
    return VocabularyTableGeneratorInfo(
        dictionary_entry=table.name,
        variable_name=f"{table.name.lower()}_vocab",
        table_name=table.name,
    )


def _generate_vocabulary_table(
    table: Table,
    engine: Engine,
    overwrite_files: bool = False,
    compress: bool = False,
) -> None:
    """Pull data out of the source database to make a vocabulary YAML file."""
    yaml_file_name: str = table.fullname + ".yaml"
    if compress:
        yaml_file_name += ".gz"
    if Path(yaml_file_name).exists() and not overwrite_files:
        logger.debug("%s already exists; not overwriting", yaml_file_name)
        return
    logger.debug("Downloading vocabulary table %s", table.name)
    download_table(table, engine, yaml_file_name, compress)


def make_tables_file(db_dsn: str, schema_name: Optional[str]) -> str:
    """Construct the YAML file representing the schema."""
    engine = get_sync_engine(create_db_engine(db_dsn, schema_name=schema_name))

    metadata = MetaData()
    metadata.reflect(engine)
    meta_dict = metadata_to_dict(metadata, schema_name, engine)

    return yaml.dump(meta_dict)


class DbConnection:
    """A connection to a database."""

    def __init__(self, engine: MaybeAsyncEngine) -> None:
        """
        Initialise an unopened database connection.

        Could be synchronous or asynchronous.
        """
        self._engine = engine
        self._connection: Connection | AsyncConnection

    async def __aenter__(self) -> Self:
        """Enter the ``with`` section, opening a connection."""
        if isinstance(self._engine, AsyncEngine):
            self._connection = await self._engine.connect()
        else:
            self._connection = self._engine.connect()
        return self

    async def __aexit__(
        self,
        _type: Optional[Type[BaseException]],
        _value: Optional[BaseException],
        _tb: Optional[TracebackType],
    ) -> None:
        """Exit the ``with`` section, closing the connection."""
        if isinstance(self._connection, AsyncConnection):
            await self._connection.close()
        else:
            self._connection.close()

    async def execute_raw_query(self, query: Executable) -> CursorResult:
        """Execute the query on the owned connection."""
        if isinstance(self._connection, AsyncConnection):
            return await self._connection.execute(query)
        return self._connection.execute(query)

    async def table_row_count(self, table_name: str) -> int:
        """Count the number of rows in the named table."""
        with await self.execute_raw_query(
            text(f"SELECT COUNT(*) FROM {table_name}")
        ) as result:
            return int(result.scalar_one())

    async def execute_query(self, query_block: Mapping[str, Any]) -> Any:
        """Execute query in query_block."""
        logger.debug("Executing query %s", query_block["name"])
        query = text(query_block["query"])
        raw_result = await self.execute_raw_query(query)

        if "dp-query" in query_block:
            result_df = pd.DataFrame(raw_result.mappings())
            logger.debug("Executing dp-query for %s", query_block["name"])
            dp_query = query_block["dp-query"]
            snsql_metadata = {"": {"": {"query_result": query_block["snsql-metadata"]}}}
            privacy = snsql.Privacy(
                epsilon=query_block["epsilon"], delta=query_block["delta"]
            )
            reader = snsql.from_df(result_df, privacy=privacy, metadata=snsql_metadata)
            private_result = reader.execute(dp_query)
            header = tuple(str(x) for x in private_result[0])
            final_result = [dict(zip(header, row)) for row in private_result[1:]]
        else:
            final_result = [
                {str(k): v for k, v in row.items()}
                for row in raw_result.mappings().fetchall()
            ]
        return final_result


def fix_type(value: Any) -> Any:
    """Make this value suitable for yaml output."""
    if isinstance(value, decimal.Decimal):
        return float(value)
    return value


def fix_types(dics: list[dict]) -> list[dict]:
    """Make all the items in this list suitable for yaml output."""
    return [{k: fix_type(v) for k, v in dic.items()} for dic in dics]


async def make_src_stats(
    dsn: str, config: Mapping, schema_name: Optional[str] = None
) -> dict[str, dict[str, Any]]:
    """
    Run the src-stats queries specified by the configuration.

    Query the src database with the queries in the src-stats block of the `config`
    dictionary, using the differential privacy parameters set in the `smartnoise-sql`
    block of `config`. Record the results in a dictionary and return it.

    :param dsn: database connection string
    :param config: a dictionary with the necessary configuration
    :param schema_name: name of the database schema
    :return: The dictionary of src-stats.
    """
    use_asyncio = config.get("use-asyncio", False)
    engine = create_db_engine(dsn, schema_name=schema_name, use_asyncio=use_asyncio)
    async with DbConnection(engine) as db_conn:
        return await make_src_stats_connection(config, db_conn)


async def make_src_stats_connection(
    config: Mapping, db_conn: DbConnection
) -> dict[str, dict[str, Any]]:
    """
    Make the ``src-stats.yaml`` file given the database connection to read from.

    :param config: configuration from ``config.yaml``.
    :param db_conn: Source database connection.
    """
    date_string = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    query_blocks = config.get("src-stats", [])
    results = await asyncio.gather(
        *[db_conn.execute_query(query_block) for query_block in query_blocks]
    )
    src_stats = {
        query_block["name"]: {
            "queries": {
                "date": date_string,
                "query": query_block["query"],
            },
            "comments": query_block.get("comments", []),
            "results": fix_types(result),
        }
        for query_block, result in zip(query_blocks, results)
    }

    for name, result in src_stats.items():
        if not result["results"]:
            logger.debug("src-stats query %s returned no results", name)

    return src_stats
