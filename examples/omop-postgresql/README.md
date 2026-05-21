# How to run datafaker process on omop schema

1. Make a YAML file representing the tables in the schema

`poetry run datafaker make-tables --orm-file ./orm.yaml --config-file ./config.yaml`

1. Interactively set generators for column data. 
`poetry run datafaker configure-generators --orm-file ./orm.yaml --config-file ./config.yaml`

1. Create schema from the ORM YAML file

`poetry run datafaker create-tables --orm-file ./orm.yaml --config-file ./config.yaml`

1. Create generator table

`poetry run datafaker create-generators --orm-file ./orm.yaml --config-file ./config.yaml --df-file ./df.py`

1. Create data

`poetry run datafaker create-data --orm-file ./orm.yaml --config-file ./config.yaml --df-file ./df.py`

1. Remove data

`poetry run datafaker remove-data --orm-file ./orm.yaml --config-file ./config.yaml`
