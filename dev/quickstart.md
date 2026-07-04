 # Quickstart (10 minutes)

## Clone and enter the repo
1. git clone https://github.com/SAFEHR-data/datafaker.git
2. cd datafaker

## Create and activate a dev environment
 
### Using Poetry:
1. poetry install --all-extras
2. poetry shell

### Without Poetry (example):
#### Linux/macOS
1. python -m venv .venv
2. source .venv/bin/activate  
3. pip install -e .[all-extras]
#### Windows
1. python -m venv .venv
2. .\.venv\Scripts\activate
3. pip install -e .[all-extras]

## Run tests
1. python -m unittest discover --verbose tests/

## Skipped tests

If you do not have psql installed the tests that require it will be automatically skipped.