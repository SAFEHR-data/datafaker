is I

Quickstart (10 minutes)

1) Clone and switch to the mssql branch
• git clone https://github.com/SAFEHR-data/datafaker.git
• cd datafaker
• git fetch origin mssql
• git checkout mssql

2) Start MSSQL locally with Docker
• docker compose up -d

This starts SQL Server 2022 on port 1433 with a preset SA password. The compose file also includes a healthcheck that waits until the server is ready.

3) Create and activate a dev environment
Using Poetry:
• poetry install --all-extras
• poetry shell

Without Poetry (example):
• python -m venv .venv
• source .venv/bin/activate  # Linux/macOS
  or .\.venv\Scripts\activate  # Windows
• pip install -e .[all-extras]

4) Run unit tests
• python -m unittest discover --verbose tests/

Configuration
Dockerized SQL Server

docker-compose.yml defines a single service:

• Image: mcr.microsoft.com/mssql/server:2022-latest
• Port mapping: 1433:1433
• Credentials:
  - User: sa
  - Password: Datafaker!Test123
• Healthcheck: uses sqlcmd to wait for readiness

You can connect with tools using:
• Host: localhost
• Port: 1433
• User: sa
• Password: Datafaker!Test123
• Database: create as needed in your workflow

Security note: The SA password in compose is for local development only. Do not reuse it in production.

Development Environment

Follow CONTRIBUTING.md for local setup:

• Create a Python virtual environment and install dependencies. With Poetry:
  - poetry install --all-extras
  - pre-commit install  # optional: if you use git hooks mentioned in contributing
  - poetry shell

• Formatting/linting: pre-commit hooks may run on commit if you installed them:
  - pre-commit run --all-files (to run manually)

Running the Application/Workflows

This project’s primary function on this branch is to support generating and testing synthetic data against MSSQL. Typical workflow:

1) Ensure the DB is up
• docker compose ps
• docker compose logs mssql  # optional
• The healthcheck should report healthy within ~1–2 minutes on first start.

2) Prepare a database/schema
• Use sqlcmd or your preferred tool to create a target database:
  - sqlcmd -S localhost -U sa -P "Datafaker!Test123" -Q "CREATE DATABASE datafakerdev;"
• Apply any required schema or run your local migration approach (if present in your branch/workflow).

3) Run your data generation scripts
• If the repo includes CLI entry points or scripts for generation (e.g., under a src/ package), execute them from your active virtual environment.
• For ad-hoc exploration, open a Python REPL inside your env and import project modules.

Note: The exact CLI entry (if any) and module names can vary while the project evolves. Check docs/source/ and src/ for the latest commands and examples.

Testing
• Run unit tests:
  - python -m unittest discover --verbose tests/

• Typical expectations:
  - Tests should pass locally once dependencies are installed.
  - If any tests require a live database, ensure docker compose is up and the DB is healthy before running.

• Coverage (if configured):
  - If coverage tooling is added later, follow the instructions in the repo (e.g., coverage run -m unittest ...).

Verifying the Setup
• Health check of MSSQL:
  - sqlcmd -S localhost -U sa -P "Datafaker!Test123" -Q "SELECT 1;"
  - Expected result: a single row containing 1.

• Minimal end-to-end sanity:
  - Create a temporary test database, run a simple table create + insert, and query it back using your scripts or sqlcmd.

Example:
• sqlcmd -S localhost -U sa -P "Datafaker!Test123" -Q "CREATE DATABASE dfsanity;"
• sqlcmd -S localhost -d df_sanity -U sa -P "Datafaker!Test123" -Q "CREATE TABLE t(id INT PRIMARY KEY, v NVARCHAR(50)); INSERT INTO t VALUES (1, 'ok'); SELECT COUNT(*) FROM t;"

Troubleshooting
• SQL Server container not healthy
  - Give it up to two minutes on cold start.
  - Ensure no conflicting service is bound to port 1433.
  - Restart: docker compose restart mssql

• Authentication errors
  - Verify the SA password exactly matches the compose file.
  - Some clients default to Windows auth; force SQL auth with user/password.

• Tests can’t find modules
  - Re-activate your virtual environment.
  - Ensure you installed the package in editable/development mode via Poetry or pip.

• SSL/Trust issues with sqlcmd 18
  - Newer sqlcmd defaults may require -C/-No flags or trust configs; the compose healthcheck uses -No. If you invoke sqlcmd manually, add -C for encrypted connections or follow your client’s guidance.

Clean Up
• Stop and remove containers: docker compose down
• Remove volumes (if you add volumes later): docker compose down -v
• Remove local virtual environment: rm -rf .venv (or delete via your OS tools)

Contributing Workflow
• Create a feature branch from mssql
• Install dev dependencies (Poetry recommended)
• Optionally enable and run pre-commit hooks before committing
• Write or update tests under tests/
• Run: python -m unittest discover --verbose tests/
• Open a pull request

Reference Commands
• Bring up MSSQL:
  - docker compose up -d

• Check health:
  - docker compose ps
  - docker compose logs mssql

• Connect with sqlcmd:
  - sqlcmd -S localhost -U sa -P "Datafaker!Test123" -Q "SELECT 1;"

• Run tests:
  - python -m unittest discover --verbose tests/

Notes
• Documentation: The README points to docs/source/ for deeper reference, design notes, and API-level details.
• The mssql branch focuses on SQL Server; other branches or future changes may differ in configuration and supported databases.