# How to run datafaker process on OMOP schema with MS SQL Server

## About

This example is for testing datafaker against a Microsoft SQL Server (MS-SQL) database using the OMOP schema. It can be used to verify that datafaker correctly generates and manages synthetic data in an MS-SQL environment.

## Challenges for MS-SQL support

Datafaker was built with PostgreSQL as its primary target. The following issues need to be addressed to support MS-SQL (Microsoft SQL Server).

### 1. Driver dependency ([#93](https://github.com/SAFEHR-data/datafaker/issues/93))

`psycopg2` (PostgreSQL driver) is a hard dependency imported directly in [datafaker/utils.py](../../datafaker/utils.py). MS-SQL requires a different driver such as `pyodbc` or `pymssql`, which are not currently listed in `pyproject.toml`. The `asyncpg` async driver is also PostgreSQL-specific; MS-SQL async support would require `aioodbc` or similar.

### 2. Hardcoded async connection string rewriting ([#94](https://github.com/SAFEHR-data/datafaker/issues/94))

In [datafaker/utils.py:208](../../datafaker/utils.py), the async DSN is built by string-replacing `postgresql://` with `postgresql+asyncpg://`. This logic would silently fail (or produce a malformed DSN) for an `mssql://` connection string.

### 3. PostgreSQL `search_path` for schema selection ([#95](https://github.com/SAFEHR-data/datafaker/issues/95))

When a schema name is provided, the code issues `SET search_path TO <schema>` via a connection-level event listener ([datafaker/utils.py:222, 305](../../datafaker/utils.py)). This is PostgreSQL-specific syntax. MS-SQL uses two-part `[schema].[table]` naming and does not support `SET search_path`. SQLAlchemy's `schema` argument on `MetaData` and `Table` objects is the correct cross-dialect approach.

### 4. PostgreSQL-specific column types in the type parser ([#96](https://github.com/SAFEHR-data/datafaker/issues/96), commit [76fec75](https://github.com/SAFEHR-data/datafaker/commit/76fec75))

[datafaker/serialize_metadata.py](../../datafaker/serialize_metadata.py) registers parsers for several PostgreSQL-only types that have no direct MS-SQL equivalent:

| PostgreSQL type | MS-SQL equivalent / issue |
|---|---|
| `postgresql.TSVECTOR` | No native equivalent (full-text indexing works differently) |
| `postgresql.CIDR` | No native network address type |
| `postgresql.BYTEA` | Use `VARBINARY(MAX)` |
| `postgresql.ARRAY` | Not supported; would need denormalisation or JSON |
| `postgresql.ENUM` | Implemented via `CHECK` constraint or lookup table |
| `postgresql.DOMAIN` | Not supported |
| `postgresql.BIT` | MS-SQL `BIT` is boolean (0/1 only); multi-bit columns use `BINARY` |
| `postgresql.REAL` / `TIMESTAMP` / `TIME` with timezone | Need MS-SQL dialect equivalents (`DATETIMEOFFSET` for tz-aware timestamps) |

### 5. `SERIAL` autoincrement columns ([#97](https://github.com/SAFEHR-data/datafaker/issues/97), commit [da4a69b](https://github.com/SAFEHR-data/datafaker/commit/da4a69b))

PostgreSQL uses `SERIAL` for autoincrement columns. The code already strips `SERIAL` for DuckDB in [datafaker/create.py:29](../../datafaker/create.py), but there is no equivalent handler for MS-SQL, which uses `IDENTITY(1,1)`.

### 6. `postgresql.UUID` type mapping ([#98](https://github.com/SAFEHR-data/datafaker/issues/98), commit [76fec75](https://github.com/SAFEHR-data/datafaker/commit/76fec75))

[datafaker/make.py:384](../../datafaker/make.py) maps `postgresql.UUID` to a generator. MS-SQL uses `UNIQUEIDENTIFIER` for UUIDs, which SQLAlchemy exposes as `sqlalchemy.dialects.mssql.UNIQUEIDENTIFIER`.

### 7. PostgreSQL-specific error handling

[datafaker/utils.py:651](../../datafaker/utils.py) catches `psycopg2.errors.UndefinedObject` to handle missing constraints gracefully. This is a PostgreSQL/psycopg2-specific exception. For MS-SQL the equivalent `pyodbc` error would need to be caught instead, or the check should be made dialect-agnostic.

### 8. `autocommit` handling in `set_db_settings`

[datafaker/utils.py:297-309](../../datafaker/utils.py) toggles `connection.autocommit` directly on the DBAPI connection before executing `SET` commands. The availability and behaviour of `autocommit` differs between `psycopg2`, `pyodbc`, and `pymssql`, so this would need testing or an abstraction per driver.

## Steps

1. Make a YAML file representing the tables in the schema

`poetry run datafaker make-tables --orm-file ./examples/omop-mssql/orm.yaml`

1. Create schema from the ORM YAML file

`poetry run datafaker create-tables --orm-file ./examples/omop-mssql/orm.yaml --config-file ./examples/omop-mssql/config.yaml`

1. Create generator table

`poetry run datafaker create-generators --orm-file ./examples/omop-mssql/orm.yaml --config-file ./examples/omop-mssql/config.yaml --df-file ./examples/omop-mssql/df.py`

1. Create data

`poetry run datafaker create-data --orm-file ./examples/omop-mssql/orm.yaml --config-file ./examples/omop-mssql/config.yaml --df-file ./examples/omop-mssql/df.py`

1. Remove data

`poetry run datafaker remove-data --orm-file ./examples/omop-mssql/orm.yaml --config-file ./examples/omop-mssql/config.yaml`
