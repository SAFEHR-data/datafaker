# How to Develop for DATAFAKER

## Pre-requisites

Please install the following software on your workstation:

1. [Poetry](https://python-poetry.org/docs/#installation).
1. [Pre-commit](https://pre-commit.com/#install).
1. [PostgreSQL](https://postgresapp.com).

## Setting up your development environment

1. Clone the GitHub repository:

    ```bash
    git clone https://github.com/SAFEHR-data/datafaker
    ```

1. In the directory of your local copy, create a virtual environment with all `datafaker` dependencies:

    ```bash
    cd datafaker
    poetry install --all-extras
    ```

    *If you don't need to [build the project documentation](#building-documentation-locally), the `--all-extras` option can be omitted.*

1. Install the git hook scripts. They will run whenever you perform a commit:

    ```bash
    pre-commit install --install-hooks
    ```

    *To execute the hooks before a commit, run `pre-commit run --all-files`.*

1. Finally, activate the Poetry shell. Now you're ready to play with the code:

    ```bash
    poetry shell
    ```

## Running unit tests

Executing unit tests is straightforward. For example inside a Poetry shell you can run:

```bash
pytest -rs
```

or outside:

```bash
poetry run pytest -rs
```

We can run tests matching a certain string with the `-k` flag:

```bash
poetry run pytest -rs -k workflow
```

### Running MsSQL tests

You need to start an MS SQL Server (or similar) and set up the `MSSQL_TEST_DSN`
environment variable to point to the server. The default for `MSSQL_TEST_DSN`
points to the docker container that `docker-compose.yaml` sets up, so the simplest
way to run the MsSQL tests is:

```bash
docker compose up -d
poetry run pytest -rs -m mssql
docker compose down
```

The `-m mssql` parameter selects only the MsSQL tests, you can omit it if you want to
run all the tests including MsSQL tests.

### Running the Snowflake tests

You need access to a Snowflake instance, and then you can set up
`SNOWFLAKE_TEST_DSN` to point to it. There is no free way to run a Snowflake
compatible instance, sadly. To run only the Snowflake tests we can use the
`-m snowflake` argument:

```bash
SNOWFLAKE_TEST_DSN="snowflake://NAME%40DOMAIN@ACCOUNT/DB?warehouse=WAREHOUSE&role=ROLE&authenticator=externalbrowser" poetry run pytest -rs -m snowflake
```

For some values of `NAME`, `DOMAIN`, `ACCOUNT`, `WAREHOUSE` and `ROLE`.

## Building documentation locally

```bash
cd docs
make html
```

*WARNING: Some systems [won't be able to import the `sphinxcontrib.napoleon` extension](https://github.com/sphinx-doc/sphinx/issues/10378). In that case,
please replace `sphinxcontrib.napoleon` with `sphinx.ext.napoleon` in `docs/source/conf.py`.*
