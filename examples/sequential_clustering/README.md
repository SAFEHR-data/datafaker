Note to self:
Use embedded python 3.12.7

1. ` make-table` Make a YAML file representing the tables in the schema.

```yaml
dsn: postgresql://username:***@host:port/uds?options=-csearch_path%3Dschema_name
schema: null
tables:
  care_site:
    columns:
      care_site_id:
        nullable: false
        primary: true
        type: INTEGER
```

```sh
poetry run datafaker make-tables --orm-file examples\sequential_clustering\outputs\orm.yaml --force
```

2.  `configure-tables` Interactively set tables to ignored, vocabulary or primary private.

This generates config.yaml

```yaml
tables:
  care_site:
    ignore: true
```

```sh
poetry run datafaker configure-tables --orm-file examples\sequential_clustering\outputs\orm.yaml --config-file examples\sequential_clustering\outputs\config.yaml
```

3. `configure-generators` Interactively set generators for column data.

This sets up the src stats in the config.yaml file.

```yaml
src-stats:
- comments:
  - The values and their counts that appear more than 7 times in column ethnicity_concept_id,
    out of a random sample of 500 rows of table person
  name: auto__person__ethnicity_concept_id
  query: SELECT value, count FROM (SELECT value, COUNT(value) AS count FROM (SELECT
    ethnicity_concept_id AS value FROM person WHERE ethnicity_concept_id IS NOT NULL
    ORDER BY RANDOM() LIMIT 500) AS _inner GROUP BY value ORDER BY count DESC) AS
    _inner WHERE 7 < count
```

```sh
poetry run datafaker configure-generators --orm-file examples\sequential_clustering\outputs\orm.yaml --config-file examples\sequential_clustering\outputs\config.yaml
```

4. `make-stats` Compute summary statistics from the source database.

```sh
poetry run datafaker make-stats --orm-file examples\sequential_clustering\outputs\orm.yaml --config-file examples\sequential_clustering\outputs\config.yaml  --stats-file examples\sequential_clustering\outputs\src-stats.yaml
```

5. `create-tables` Create schema from the ORM YAML file.

```sh
poetry run datafaker create-tables --orm-file examples\sequential_clustering\outputs\orm.yaml --config-file examples\sequential_clustering\outputs\config.yaml
```

6. `create-generators`     Make a datafaker file of generator classes.  

```sh
poetry run datafaker create-generators --orm-file examples\sequential_clustering\outputs\orm.yaml --config-file examples\sequential_clustering\outputs\config.yaml --stats-file examples\sequential_clustering\outputs\src-stats.yaml
```


7. `create-data`           Populate the schema in the target directory with synthetic data. 
```sh
poetry run datafaker create-data --orm-file examples\sequential_clustering\outputs\orm.yaml --config-file examples\sequential_clustering\outputs\config.yaml --df-file examples/sequential_clustering/outputs/df.py
```

8. `remove-data` Truncate non-vocabulary tables in the destination schema.
```sh
poetry run datafaker remove-data --orm-file examples/sequential_clustering/outputs/orm.yaml --config-file examples/sequential_clustering/outputs/config.yaml
```