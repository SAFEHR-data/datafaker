# Overview
## Purpose: 
Generate synthetic data for SQL databases, based on the Alan Turing Institute’s sqlsynthgen.

## Databases Supported:
- PostgreSQL
- DuckDB
- MSSQL


## Prerequisites

- Python 3.11+
- Poetry installed (recommended) or another Python environment manager
- Docker Desktop or Docker Engine 20.10+ if you prefer running databases locally via containers
- Docker Compose v2
- Git

### Optional:
- psql or your chosen SQL client to inspect a running database
- sqlcmd or Azure Data Studio for checking the MSSQL instance
- A modern shell with make (if you plan to add convenience targets)

## Repository Layout

```text
project/
├─ datafaker /
│  ├─ interactive/
│  ├─ json_schemas/
│  ├─ proposers/
│  ├─ __init__.py
│     ...
├─ dev/
├─ docs/
├─ examples/
├─ tests/
├─ README.md
├─ CONTRIBUTING.md
├─ poetry.lock
└─ pyproject.toml
```

