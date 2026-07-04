.

Quickstart (10 minutes)

1) Clone and enter the repo
• git clone https://github.com/SAFEHR-data/datafaker.git
• cd datafaker

2) Create and activate a dev environment
Using Poetry:
• poetry install --all-extras
• poetry shell

Without Poetry (example):
• python -m venv .venv
• source .venv/bin/activate  # Linux/macOS
  or .\.venv\Scripts\activate  # Windows
• pip install -e .[all-extras]

3) Run tests
• python -m unittest discover --verbose tests/

If tests or examples require a database, follow the “Databases” section below before running them.

Configuration
• Environment variables: The project generally does not require secrets for unit tests. For integration examples, supply DB connection variables in your shell or a local .env you do not commit.
• Docs: See docs/source/ for in-depth usage, configuration patterns, and engine-specific notes.

Development Environment

Recommended workflow from CONTRIBUTING.md:

• Install dependencies with Poetry:
  - poetry install --all-extras
  - pre-commit install  # optional, to enable formatting/lint on commit
  - poetry shell

• Run linters/formatters via pre-commit hooks when installed:
  - pre-commit run --all-files

Running the Application/Workflows

This repository provides Python libraries and utilities to synthesize data. On main, typical development flows are library-centric rather than daemon-or-service oriented.

Examples:
• Open a Python REPL in your environment and import project modules to experiment.
• Create a short script under examples/ or your own folder that:
  - connects to a target database (e.g., PostgreSQL via SQLAlchemy/psycopg)
  - generates tables and rows using project APIs
  - inserts them into your database

Because the codebase evolves, check docs/source/ and src/ for up-to-date module/CLI entry points. If a CLI is present, invoke it from your virtual environment and pass connection parameters as flags or env vars.

Databases (optional, for integration testing)

Unit tests should run without a live database. If you want to validate end-to-end behavior:

• PostgreSQL locally with Docker:
  - docker run --name df-postgres -e POSTGRESPASSWORD=devpass -p 5432:5432 -d postgres:16
  - psql -h localhost -U postgres -c "CREATE DATABASE datafakerdev;"  # set PGPASSWORD=devpass in your shell

• Connection variables (example):
  - DBHOST=localhost
  - DBPORT=5432
  - DBNAME=datafakerdev
  - DBUSER=postgres
  - DBPASSWORD=devpass

• Sanity check:
  - psql -h localhost -U postgres -d datafakerdev -c "SELECT 1;"

Adjust versions and credentials to your local policies.

Testing
• Run unit tests:
  - python -m unittest discover --verbose tests/

• If coverage is configured in the project, use:
  - coverage run -m unittest discover --verbose tests/
  - coverage report -m

• If any tests depend on a database, ensure it is running and environment variables are set accordingly. Otherwise, unit tests should pass without DB services.

Verifying the Setup
• Import check:
  - python -c "import datafaker; print('ok')"
  - Expected: prints ok with no ImportError.

• Minimal end-to-end sketch (pseudo-steps):
  - Start a local PostgreSQL (optional).
  - From a Python REPL, generate a small table and write a few rows.
  - Query back using your SQL client to confirm inserts.

Troubleshooting
• Cannot import modules
  - Ensure virtual environment is active.
  - Reinstall dependencies: poetry install or pip install -e .[all-extras].

• Pre-commit not found
  - poetry run pre-commit install or pip install pre-commit, then re-run pre-commit run --all-files.

• Database connection errors (optional flow)
  - Verify host/port, user, password, and DB existence.
  - Confirm container is healthy if using Docker.

• Version mismatches
  - Confirm your Python version is 3.11+.
  - If using system Python, prefer a clean virtual environment to avoid package conflicts.

Clean Up
• Deactivate environment: exit the Poetry shell or run deactivate for venv.
• Remove local venv: rm -rf .venv (or your tool’s equivalent).
• If you started containers: docker rm -f df-postgres (or docker compose down).

Contributing Workflow
• Create a feature branch from main
• Install dev dependencies (Poetry recommended)
• Optionally enable and run pre-commit hooks before committing
• Write or update tests under tests/
• Run: python -m unittest discover --verbose tests/
• Open a pull request

Reference Commands
• Install (Poetry):
  - poetry install --all-extras

• Activate shell:
  - poetry shell

• Lint/format (optional):
  - pre-commit install
  - pre-commit run --all-files

• Run tests:
  - python -m unittest discover --verbose tests/

• Optional PostgreSQL with Docker:
  - docker run --name df-postgres -e POSTGRESPASSWORD=devpass -p 5432:5432 -d postgres:16
  - psql -h localhost -U postgres -d postgres -c "CREATE DATABASE datafaker_dev;"

Notes
• The README directs you to docs/source/ for deeper reference and API-level details.
• The main branch focuses on the core, database-agnostic library behavior; engine-specific setup may live in separate docs or branches.