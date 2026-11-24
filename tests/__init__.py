"""Test module in which we can set up and tear down over all tests."""
import os
from collections.abc import Iterable
from typing import Any
from sqlalchemy import create_engine, insert, text, Column, Engine, MetaData, Table
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema, DropSchema

_snowflake_engine: Engine | None = None


def get_snowflake_dsn(schema: str="") -> str:
    """Get a Snowflake DSN, or None if the environment variables are not set."""
    db = os.environ.get("SNOWFLAKE_DATABASE")
    role = os.environ.get("SNOWFLAKE_ROLE")
    wh = os.environ.get("SNOWFLAKE_WAREHOUSE")
    user = os.environ.get("SNOWFLAKE_USER")
    if db and role and wh and user:
        return (
            f"snowflake://{user}@ATKJNCU-LDS_DEV/{db}/{schema}?"
            f"warehouse={wh}&role={role}&authenticator=externalbrowser"
        )
    return None


def create_snowflake_schema(schema: str) -> bool:
    """Create a schema."""
    return execute_snowflake([CreateSchema(schema)])


def drop_snowflake_schema(schema: str) -> bool:
    """Destroy a schema."""
    return execute_snowflake([DropSchema(schema)])


def execute_snowflake_sql(sqls: Iterable[str]) -> bool:
    """Execute SQL statements on the test Snowflake database."""
    return execute_snowflake(text(sql) for sql in sqls)


def execute_snowflake(commands: Iterable[Any]) -> bool:
    """Execute commands on the test Snowflake database."""
    global _snowflake_engine
    if not _snowflake_engine:
        return False
    print("creating session")
    n = 0
    with Session(_snowflake_engine, autoflush=False) as session:
        for command in commands:
            try:
                session.execute(command)
                # or session.execute(insert(table), [{"col1": c1, "col2": c2}])
            except Exception as e:
                breakpoint()
            print(n)
            n += 1
        print("end")
    print("Written")
    return True


def execute_snowflake_bulk_insert(table_name: str, data: list[dict[str, Any]]) -> bool:
    """Insert the data specified into the table specified."""
    global _snowflake_engine
    if not _snowflake_engine or not data:
        return False
    metadata = MetaData()
    metadata.reflect(_snowflake_engine)
    print(f"Writing data to table {table_name}")
    with Session(_snowflake_engine) as session:
        session.execute(insert(metadata.tables[table_name]), data)
        session.commit()
    print("Written")


def setUpModule():
    """Initialise the Snowflake connection if needed."""
    global _snowflake_engine
    dsn = get_snowflake_dsn("")
    if dsn is None:
        _snowflake_engine = None
    else:
        _snowflake_engine = create_engine(dsn, paramstyle="qmark", connect_args={
            "paramstyle": "qmark",
            "CLIENT_STAGE_ARRAY_BINDING_THRESHOLD": 65536,
        })


def tearDownModule():
    """Deinitialize Snowflake."""
    global _snowflake_engine
    if _snowflake_engine != None:
        _snowflake_engine.dispose()
        _snowflake_engine = None
