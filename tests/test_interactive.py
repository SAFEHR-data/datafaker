""" Tests for the base module. """
import copy
import random
import re
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

from sqlalchemy import insert, select

from datafaker.generators import NullPartitionedNormalGeneratorFactory
from datafaker.interactive import (
    DbCmd,
    GeneratorCmd,
    MissingnessCmd,
    TableCmd,
    update_config_generators,
)
from tests.utils import GeneratesDBTestCase, RequiresDBTestCase


class TestDbCmdMixin(DbCmd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.messages: list[tuple[str, list, dict[str, any]]] = []
        self.headings: list[str] = []
        self.rows: list[list[str]] = []
        self.column_items: list[str] = []
        self.columns: dict[str, list[str]] = {}

    def print(self, text: str, *args, **kwargs):
        self.messages.append((text, args, kwargs))

    def print_table(self, headings: list[str], rows: list[list[str]]):
        self.headings = headings
        self.rows = rows

    def print_table_by_columns(self, columns: dict[str, list[str]]):
        self.columns = columns

    def columnize(self, items: list[str]):
        self.column_items.append(items)

    def ask_save(self) -> str:
        return "yes"


class TestTableCmd(TableCmd, TestDbCmdMixin):
    """TableCmd but mocked"""


class ConfigureTablesTests(RequiresDBTestCase):
    """Testing configure-tables."""

    def _get_cmd(self, config) -> TestTableCmd:
        return TestTableCmd(self.dsn, self.schema_name, self.metadata, config)


class ConfigureTablesSrcTests(ConfigureTablesTests):
    """Testing configure-tables with src.dump."""

    dump_file_path = "src.dump"
    database_name = "src"
    schema_name = "public"

    def test_table_name_prompts(self) -> None:
        """Test that the prompts follow the names of the tables."""
        config = {}
        with self._get_cmd(config) as tc:
            table_names = list(self.metadata.tables.keys())
            for t in table_names:
                self.assertIn(t, tc.prompt)
                tc.do_next("")
            self.assertListEqual(tc.messages, [(TableCmd.INFO_NO_MORE_TABLES, (), {})])
            tc.reset()
            for t in reversed(table_names):
                self.assertIn(t, tc.prompt)
                tc.do_previous("")
            self.assertListEqual(
                tc.messages, [(TableCmd.ERROR_ALREADY_AT_START, (), {})]
            )
            tc.reset()
            bad_table_name = "notarealtable"
            tc.do_next(bad_table_name)
            self.assertListEqual(
                tc.messages, [(TableCmd.ERROR_NO_SUCH_TABLE, (bad_table_name,), {})]
            )
            tc.reset()
            good_table_name = table_names[2]
            tc.do_next(good_table_name)
            self.assertListEqual(tc.messages, [])
            self.assertIn(good_table_name, tc.prompt)

    def test_column_display(self) -> None:
        """Test that we can see the names of the columns."""
        config = {}
        with self._get_cmd(config) as tc:
            tc.do_next("unique_constraint_test")
            tc.do_columns("")
            self.assertListEqual(
                tc.rows,
                [
                    ["id", "INTEGER", True, False, ""],
                    ["a", "BOOLEAN", False, False, ""],
                    ["b", "BOOLEAN", False, False, ""],
                    ["c", "TEXT", False, False, ""],
                ],
            )

    def test_null_configuration(self) -> None:
        """A table still works if its configuration is None."""
        config = {
            "tables": None,
        }
        with self._get_cmd(config) as tc:
            tc.do_next("unique_constraint_test")
            tc.do_private("")
            tc.do_quit("")
            tables = tc.config["tables"]
            self.assertFalse(
                tables["unique_constraint_test"].get("vocabulary_table", False)
            )
            self.assertFalse(tables["unique_constraint_test"].get("ignore", False))
            self.assertTrue(
                tables["unique_constraint_test"].get("primary_private", False)
            )

    def test_null_table_configuration(self) -> None:
        """A table still works if its configuration is None."""
        config = {
            "tables": {
                "unique_constraint_test": None,
            },
        }
        with self._get_cmd(config) as tc:
            tc.do_next("unique_constraint_test")
            tc.do_private("")
            tc.do_quit("")
            tables = tc.config["tables"]
            self.assertFalse(
                tables["unique_constraint_test"].get("vocabulary_table", False)
            )
            self.assertFalse(tables["unique_constraint_test"].get("ignore", False))
            self.assertTrue(
                tables["unique_constraint_test"].get("primary_private", False)
            )

    def test_configure_tables(self) -> None:
        """Test that we can change columns to ignore, vocab or generate."""
        config = {
            "tables": {
                "unique_constraint_test": {
                    "vocabulary_table": True,
                },
                "no_pk_test": {
                    "ignore": True,
                },
                "hospital_visit": {
                    "num_passes": 0,
                },
                "empty_vocabulary": {
                    "private": True,
                },
            },
        }
        with self._get_cmd(config) as tc:
            tc.do_next("unique_constraint_test")
            tc.do_generate("")
            tc.do_next("person")
            tc.do_vocabulary("")
            tc.do_next("mitigation_type")
            tc.do_ignore("")
            tc.do_next("hospital_visit")
            tc.do_private("")
            tc.do_quit("")
            tc.do_next("empty_vocabulary")
            tc.do_empty("")
            tc.do_quit("")
            tables = tc.config["tables"]
            self.assertFalse(
                tables["unique_constraint_test"].get("vocabulary_table", False)
            )
            self.assertFalse(tables["unique_constraint_test"].get("ignore", False))
            self.assertFalse(
                tables["unique_constraint_test"].get("primary_private", False)
            )
            self.assertEqual(tables["unique_constraint_test"].get("num_passes", 1), 1)
            self.assertFalse(tables["no_pk_test"].get("vocabulary_table", False))
            self.assertTrue(tables["no_pk_test"].get("ignore", False))
            self.assertFalse(tables["no_pk_test"].get("primary_private", False))
            self.assertEqual(tables["no_pk_test"].get("num_rows_per_pass", 1), 1)
            self.assertTrue(tables["person"].get("vocabulary_table", False))
            self.assertFalse(tables["person"].get("ignore", False))
            self.assertFalse(tables["person"].get("primary_private", False))
            self.assertEqual(tables["person"].get("num_rows_per_pass", 1), 1)
            self.assertFalse(tables["mitigation_type"].get("vocabulary_table", False))
            self.assertTrue(tables["mitigation_type"].get("ignore", False))
            self.assertFalse(tables["mitigation_type"].get("primary_private", False))
            self.assertEqual(tables["mitigation_type"].get("num_rows_per_pass", 1), 1)
            self.assertFalse(tables["hospital_visit"].get("vocabulary_table", False))
            self.assertFalse(tables["hospital_visit"].get("ignore", False))
            self.assertTrue(tables["hospital_visit"].get("primary_private", False))
            self.assertEqual(tables["hospital_visit"].get("num_rows_per_pass", 1), 1)
            self.assertFalse(tables["empty_vocabulary"].get("vocabulary_table", False))
            self.assertFalse(tables["empty_vocabulary"].get("ignore", False))
            self.assertFalse(tables["empty_vocabulary"].get("primary_private", False))
            self.assertEqual(tables["empty_vocabulary"].get("num_rows_per_pass", 1), 0)

    def test_print_data(self) -> None:
        """Test that we can print random rows from the table and random data from columns."""
        person_table = self.metadata.tables["person"]
        with self.engine.connect() as conn:
            person_rows = conn.execute(select(person_table)).mappings().fetchall()
            person_data = {row["person_id"]: row for row in person_rows}
            name_set = {row["name"] for row in person_rows}
        person_headings = ["person_id", "name", "research_opt_out", "stored_from"]
        with self._get_cmd({}) as tc:
            tc.do_next("person")
            tc.do_data("")
            self.assertListEqual(tc.headings, person_headings)
            self.assertEqual(len(tc.rows), 10)  # default number of rows is 10
            for row in tc.rows:
                expected = person_data[row[0]]
                self.assertListEqual(row, [expected[h] for h in person_headings])
            tc.reset()
            rows_to_get_count = 6
            tc.do_data(str(rows_to_get_count))
            self.assertListEqual(tc.headings, person_headings)
            self.assertEqual(len(tc.rows), rows_to_get_count)
            for row in tc.rows:
                expected = person_data[row[0]]
                self.assertListEqual(row, [expected[h] for h in person_headings])
            tc.reset()
            to_get_count = 12
            tc.do_data(f"{to_get_count} name")
            self.assertEqual(len(tc.column_items), 1)
            self.assertEqual(len(tc.column_items[0]), to_get_count)
            self.assertLessEqual(set(tc.column_items[0]), name_set)
            tc.reset()
            tc.do_data(f"{to_get_count} name 12")
            self.assertEqual(len(tc.column_items), 1)
            self.assertEqual(len(tc.column_items[0]), to_get_count)
            tc.reset()
            tc.do_data(f"{to_get_count} name 13")
            self.assertEqual(len(tc.column_items), 1)
            self.assertEqual(
                set(tc.column_items[0]), set(filter(lambda n: 13 <= len(n), name_set))
            )
            tc.reset()
            tc.do_data(f"{to_get_count} name 16")
            self.assertEqual(len(tc.column_items), 1)
            self.assertEqual(
                set(tc.column_items[0]), set(filter(lambda n: 16 <= len(n), name_set))
            )

    def test_list_tables(self) -> None:
        """Test that we can list the tables"""
        config = {
            "tables": {
                "unique_constraint_test": {
                    "vocabulary_table": True,
                },
                "no_pk_test": {
                    "ignore": True,
                },
            },
        }
        with self._get_cmd(config) as tc:
            tc.do_next("unique_constraint_test")
            tc.do_ignore("")
            tc.do_next("person")
            tc.do_vocabulary("")
            tc.reset()
            tc.do_tables("")
            person_listed = False
            unique_constraint_test_listed = False
            no_pk_test_listed = False
            for text, args, kwargs in tc.messages:
                if args[2] == "person":
                    self.assertFalse(person_listed)
                    person_listed = True
                    self.assertEqual(args[0], "G")
                    self.assertEqual(args[1], "->V")
                elif args[2] == "unique_constraint_test":
                    self.assertFalse(unique_constraint_test_listed)
                    unique_constraint_test_listed = True
                    self.assertEqual(args[0], "V")
                    self.assertEqual(args[1], "->I")
                elif args[2] == "no_pk_test":
                    self.assertFalse(no_pk_test_listed)
                    no_pk_test_listed = True
                    self.assertEqual(args[0], "I")
                    self.assertEqual(args[1], "   ")
                else:
                    self.assertEqual(args[0], "G")
                    self.assertEqual(args[1], "   ")
            self.assertTrue(person_listed)
            self.assertTrue(unique_constraint_test_listed)
            self.assertTrue(no_pk_test_listed)


class ConfigureTablesInstrumentsTests(ConfigureTablesTests):
    """Testing configure-tables with the instrument.sql database."""

    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    def test_sanity_checks_both(self) -> None:
        config = {
            "tables": {
                "model": {
                    "vocabulary_table": True,
                },
                "manufacturer": {
                    "ignore": True,
                },
                "player": {
                    "num_rows_per_pass": 0,
                },
            },
        }
        with self._get_cmd(config) as tc:
            tc.reset()
            tc.do_quit("")
            self.assertEqual(tc.messages[0], (TableCmd.NOTE_TEXT_NO_CHANGES, (), {}))
            self.assertEqual(
                tc.messages[1], (TableCmd.WARNING_TEXT_PROBLEMS_EXIST, (), {})
            )
            self.assertEqual(
                tc.messages[2],
                (
                    TableCmd.WARNING_TEXT_VOCAB_TO_NON_VOCAB,
                    ("model", "manufacturer"),
                    {},
                ),
            )
            self.assertEqual(
                tc.messages[3], (TableCmd.WARNING_TEXT_POTENTIAL_PROBLEMS, (), {})
            )
            self.assertEqual(
                tc.messages[4],
                (
                    TableCmd.WARNING_TEXT_NON_EMPTY_TO_EMPTY,
                    ("signature_model", "player"),
                    {},
                ),
            )

    def test_sanity_checks_warnings_only(self) -> None:
        config = {
            "tables": {
                "model": {
                    "vocabulary_table": True,
                },
                "manufacturer": {
                    "ignore": True,
                },
                "player": {
                    "num_rows_per_pass": 0,
                },
            },
        }
        with TestTableCmd(self.dsn, self.schema_name, self.metadata, config) as tc:
            tc.do_next("manufacturer")
            tc.do_vocabulary("")
            tc.reset()
            tc.do_quit("")
            self.assertEqual(
                tc.messages[0],
                (
                    TableCmd.NOTE_TEXT_CHANGING,
                    ("manufacturer", "ignore", "vocabulary"),
                    {},
                ),
            )
            self.assertEqual(
                tc.messages[1], (TableCmd.WARNING_TEXT_POTENTIAL_PROBLEMS, (), {})
            )
            self.assertEqual(
                tc.messages[2],
                (
                    TableCmd.WARNING_TEXT_NON_EMPTY_TO_EMPTY,
                    ("signature_model", "player"),
                    {},
                ),
            )

    def test_sanity_checks_errors_only(self) -> None:
        config = {
            "tables": {
                "model": {
                    "vocabulary_table": True,
                },
                "manufacturer": {
                    "ignore": True,
                },
                "player": {
                    "num_rows_per_pass": 0,
                },
            },
        }
        with TestTableCmd(self.dsn, self.schema_name, self.metadata, config) as tc:
            tc.do_next("signature_model")
            tc.do_empty("")
            tc.reset()
            tc.do_quit("")
            self.assertEqual(
                tc.messages[0],
                (
                    TableCmd.NOTE_TEXT_CHANGING,
                    ("signature_model", "generate", "empty"),
                    {},
                ),
            )
            self.assertEqual(
                tc.messages[1], (TableCmd.WARNING_TEXT_PROBLEMS_EXIST, (), {})
            )
            self.assertEqual(
                tc.messages[2],
                (
                    TableCmd.WARNING_TEXT_VOCAB_TO_NON_VOCAB,
                    ("model", "manufacturer"),
                    {},
                ),
            )


class TestGeneratorCmd(GeneratorCmd, TestDbCmdMixin):
    """GeneratorCmd but mocked"""

    def get_proposals(self) -> dict[str, tuple[int, str, str, list[str]]]:
        """
        Returns a dict of generator name to a tuple of (index, fit_string, [list,of,samples])
        """
        return {
            kw["name"]: (kw["index"], kw["fit"], kw["sample"].split("; "))
            for (s, _, kw) in self.messages
            if s == self.PROPOSE_GENERATOR_SAMPLE_TEXT
        }


class ConfigureGeneratorsTests(RequiresDBTestCase):
    """Testing configure-generators."""

    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    def _get_cmd(self, config) -> TestGeneratorCmd:
        return TestGeneratorCmd(self.dsn, self.schema_name, self.metadata, config)

    def test_null_configuration(self) -> None:
        """Test that the tables having null configuration does not break."""
        config = {
            "tables": None,
        }
        with self._get_cmd(config) as gc:
            TABLE = "model"
            gc.do_next(f"{TABLE}.name")
            gc.do_propose("")
            gc.do_compare("")
            gc.do_set("1")
            gc.do_quit("")
            self.assertEqual(len(gc.config["tables"][TABLE]["row_generators"]), 1)

    def test_null_table_configuration(self) -> None:
        """Test that a table having null configuration does not break."""
        config = {
            "tables": {
                "model": None,
            }
        }
        with self._get_cmd(config) as gc:
            TABLE = "model"
            gc.do_next(f"{TABLE}.name")
            gc.do_propose("")
            gc.do_set("1")
            gc.do_quit("")
            self.assertEqual(len(gc.config["tables"][TABLE]["row_generators"]), 1)

    def test_prompts(self) -> None:
        """Test that the prompts follow the names of the columns and assigned generators."""
        config = {}
        with self._get_cmd(config) as gc:
            for table_name, table_meta in self.metadata.tables.items():
                for column_name, column_meta in table_meta.columns.items():
                    self.assertIn(table_name, gc.prompt)
                    self.assertIn(column_name, gc.prompt)
                    if column_meta.primary_key:
                        self.assertIn("[pk]", gc.prompt)
                    else:
                        self.assertNotIn("[pk]", gc.prompt)
                    gc.do_next("")
            self.assertListEqual(
                gc.messages, [(GeneratorCmd.INFO_NO_MORE_TABLES, (), {})]
            )
            gc.reset()
            for table_name, table_meta in reversed(list(self.metadata.tables.items())):
                for column_name, column_meta in reversed(
                    list(table_meta.columns.items())
                ):
                    self.assertIn(table_name, gc.prompt)
                    self.assertIn(column_name, gc.prompt)
                    if column_meta.primary_key:
                        self.assertIn("[pk]", gc.prompt)
                    else:
                        self.assertNotIn("[pk]", gc.prompt)
                    gc.do_previous("")
            self.assertListEqual(
                gc.messages, [(GeneratorCmd.ERROR_ALREADY_AT_START, (), {})]
            )
            gc.reset()
            bad_table_name = "notarealtable"
            gc.do_next(bad_table_name)
            self.assertListEqual(
                gc.messages,
                [(GeneratorCmd.ERROR_NO_SUCH_TABLE_OR_COLUMN, (bad_table_name,), {})],
            )
            gc.reset()

    def test_set_generator_mimesis(self) -> None:
        """Test that we can set one generator to a mimesis generator."""
        with self._get_cmd({}) as gc:
            TABLE = "model"
            COLUMN = "name"
            GENERATOR = "person.first_name"
            gc.do_next(f"{TABLE}.{COLUMN}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[f"generic.{GENERATOR}"][0]))
            gc.do_quit("")
            self.assertEqual(len(gc.config["tables"][TABLE]["row_generators"]), 1)
            self.assertDictEqual(
                gc.config["tables"][TABLE]["row_generators"][0],
                {"name": f"generic.{GENERATOR}", "columns_assigned": [COLUMN]},
            )

    def test_set_generator_distribution(self) -> None:
        """Test that we can set one generator to gaussian."""
        with self._get_cmd({}) as gc:
            TABLE = "string"
            COLUMN = "frequency"
            GENERATOR = "dist_gen.normal"
            gc.do_next(f"{TABLE}.{COLUMN}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[GENERATOR][0]))
            gc.do_quit("")
            row_gens = gc.config["tables"][TABLE]["row_generators"]
            self.assertEqual(len(row_gens), 1)
            row_gen = row_gens[0]
            self.assertEqual(row_gen["name"], GENERATOR)
            self.assertListEqual(row_gen["columns_assigned"], [COLUMN])
            self.assertDictEqual(
                row_gen["kwargs"],
                {
                    "mean": f'SRC_STATS["auto__{TABLE}"]["results"][0]["mean__{COLUMN}"]',
                    "sd": f'SRC_STATS["auto__{TABLE}"]["results"][0]["stddev__{COLUMN}"]',
                },
            )
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertSetEqual(
                set(gc.config["src-stats"][0].keys()), {"comments", "name", "query"}
            )
            self.assertEqual(gc.config["src-stats"][0]["name"], f"auto__{TABLE}")
            self.assertEqual(
                gc.config["src-stats"][0]["query"],
                f"SELECT AVG({COLUMN}) AS mean__{COLUMN}, STDDEV({COLUMN}) AS stddev__{COLUMN} FROM {TABLE}",
            )

    def test_set_generator_distribution_directly(self) -> None:
        """Test that we can set one generator to gaussian without going through propose."""
        with self._get_cmd({}) as gc:
            TABLE = "string"
            COLUMN = "frequency"
            GENERATOR = "dist_gen.normal"
            gc.do_next(f"{TABLE}.{COLUMN}")
            gc.reset()
            gc.do_set(GENERATOR)
            self.assertListEqual(gc.messages, [])
            gc.do_quit("")
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertSetEqual(
                set(gc.config["src-stats"][0].keys()), {"comments", "name", "query"}
            )
            self.assertEqual(gc.config["src-stats"][0]["name"], f"auto__{TABLE}")
            self.assertEqual(
                gc.config["src-stats"][0]["query"],
                f"SELECT AVG({COLUMN}) AS mean__{COLUMN}, STDDEV({COLUMN}) AS stddev__{COLUMN} FROM {TABLE}",
            )

    def test_set_generator_choice(self) -> None:
        """Test that we can set one generator to uniform choice."""
        with self._get_cmd({}) as gc:
            TABLE = "string"
            COLUMN = "frequency"
            GENERATOR = "dist_gen.choice"
            gc.do_next(f"{TABLE}.{COLUMN}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[GENERATOR][0]))
            gc.do_quit("")
            row_gens = gc.config["tables"][TABLE]["row_generators"]
            self.assertEqual(len(row_gens), 1)
            row_gen = row_gens[0]
            self.assertEqual(row_gen["name"], GENERATOR)
            self.assertListEqual(row_gen["columns_assigned"], [COLUMN])
            self.assertDictEqual(
                row_gen["kwargs"],
                {
                    "a": f'SRC_STATS["auto__{TABLE}__{COLUMN}"]["results"]',
                },
            )
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertSetEqual(
                set(gc.config["src-stats"][0].keys()), {"comments", "name", "query"}
            )
            self.assertEqual(
                gc.config["src-stats"][0]["name"], f"auto__{TABLE}__{COLUMN}"
            )
            self.assertEqual(
                gc.config["src-stats"][0]["query"],
                f"SELECT {COLUMN} AS value FROM {TABLE} WHERE {COLUMN} IS NOT NULL GROUP BY value ORDER BY COUNT({COLUMN}) DESC",
            )

    def test_weighted_choice_generator_generates_choices(self) -> None:
        """Test that propose and compare show weighted_choice's values."""
        with self._get_cmd({}) as gc:
            TABLE = "string"
            COLUMN = "position"
            GENERATOR = "dist_gen.weighted_choice"
            VALUES = {1, 2, 3, 4, 5, 6}
            gc.do_next(f"{TABLE}.{COLUMN}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gen_proposal = proposals[GENERATOR]
            self.assertSubset(set(gen_proposal[2]), {str(v) for v in VALUES})
            gc.do_compare(str(gen_proposal[0]))
            col_heading = f"{gen_proposal[0]}. {GENERATOR}"
            self.assertIn(col_heading, gc.columns)
            self.assertSubset(set(gc.columns[col_heading]), VALUES)

    def test_merge_columns(self) -> None:
        """Test that we can merge columns and set a multivariate generator"""
        TABLE = "string"
        COLUMN_1 = "frequency"
        COLUMN_2 = "position"
        GENERATOR_TO_DISCARD = "dist_gen.choice"
        GENERATOR = "dist_gen.multivariate_normal"
        with self._get_cmd({}) as gc:
            gc.do_next(f"{TABLE}.{COLUMN_2}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            # set a generator, but this should not exist after merging
            gc.do_set(str(proposals[GENERATOR_TO_DISCARD][0]))
            gc.do_next(f"{TABLE}.{COLUMN_1}")
            self.assertIn(TABLE, gc.prompt)
            self.assertIn(COLUMN_1, gc.prompt)
            self.assertNotIn(COLUMN_2, gc.prompt)
            gc.do_propose("")
            proposals = gc.get_proposals()
            # set a generator, but this should not exist either
            gc.do_set(str(proposals[GENERATOR_TO_DISCARD][0]))
            gc.do_previous("")
            self.assertIn(TABLE, gc.prompt)
            self.assertIn(COLUMN_1, gc.prompt)
            self.assertNotIn(COLUMN_2, gc.prompt)
            gc.do_merge(COLUMN_2)
            self.assertIn(TABLE, gc.prompt)
            self.assertIn(COLUMN_1, gc.prompt)
            self.assertIn(COLUMN_2, gc.prompt)
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[GENERATOR][0]))
            gc.do_quit("")
            row_gens = gc.config["tables"][TABLE]["row_generators"]
            self.assertEqual(len(row_gens), 1)
            row_gen = row_gens[0]
            self.assertEqual(row_gen["name"], GENERATOR)
            self.assertListEqual(row_gen["columns_assigned"], [COLUMN_1, COLUMN_2])

    def test_unmerge_columns(self) -> None:
        """Test that we can unmerge columns and generators are removed"""
        TABLE = "string"
        COLUMN_1 = "frequency"
        COLUMN_2 = "position"
        COLUMN_3 = "model_id"
        REMAINING_GEN = "gen3"
        config = {
            "tables": {
                TABLE: {
                    "row_generators": [
                        {"name": "gen1", "columns_assigned": [COLUMN_1, COLUMN_2]},
                        {"name": REMAINING_GEN, "columns_assigned": [COLUMN_3]},
                    ]
                }
            }
        }
        with self._get_cmd(config) as gc:
            gc.do_next(f"{TABLE}.{COLUMN_2}")
            self.assertIn(TABLE, gc.prompt)
            self.assertIn(COLUMN_1, gc.prompt)
            self.assertIn(COLUMN_2, gc.prompt)
            gc.do_unmerge(COLUMN_1)
            self.assertIn(TABLE, gc.prompt)
            self.assertNotIn(COLUMN_1, gc.prompt)
            self.assertIn(COLUMN_2, gc.prompt)
            # Next generator should be the unmerged one
            gc.do_next("")
            self.assertIn(TABLE, gc.prompt)
            self.assertIn(COLUMN_1, gc.prompt)
            self.assertNotIn(COLUMN_2, gc.prompt)
            gc.do_quit("")
            # Both generators should have disappeared
            row_gens = gc.config["tables"][TABLE]["row_generators"]
            self.assertEqual(len(row_gens), 1)
            row_gen = row_gens[0]
            self.assertEqual(row_gen["name"], REMAINING_GEN)
            self.assertListEqual(row_gen["columns_assigned"], [COLUMN_3])

    def test_old_generators_remain(self) -> None:
        """Test that we can set one generator and keep an old one."""
        config = {
            "tables": {
                "string": {
                    "row_generators": [
                        {
                            "name": "dist_gen.normal",
                            "columns_assigned": ["frequency"],
                            "kwargs": {
                                "mean": 'SRC_STATS["auto__string"][0]["mean__frequency"]',
                                "sd": 'SRC_STATS["auto__string"][0]["stddev__frequency"]',
                            },
                        }
                    ]
                }
            },
            "src-stats": [
                {
                    "name": "auto__string",
                    "query": "SELECT AVG(frequency) AS mean__frequency, STDDEV(frequency) AS stddev__frequency FROM string",
                }
            ],
        }
        with self._get_cmd(config) as gc:
            TABLE = "model"
            COLUMN = "name"
            GENERATOR = "person.first_name"
            gc.do_next(f"{TABLE}.{COLUMN}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[f"generic.{GENERATOR}"][0]))
            gc.do_quit("")
            self.assertEqual(len(gc.config["tables"][TABLE]["row_generators"]), 1)
            self.assertDictEqual(
                gc.config["tables"][TABLE]["row_generators"][0],
                {"name": f"generic.{GENERATOR}", "columns_assigned": [COLUMN]},
            )
            row_gens = gc.config["tables"]["string"]["row_generators"]
            self.assertEqual(len(row_gens), 1)
            row_gen = row_gens[0]
            self.assertEqual(row_gen["name"], "dist_gen.normal")
            self.assertListEqual(row_gen["columns_assigned"], ["frequency"])
            self.assertDictEqual(
                row_gen["kwargs"],
                {
                    "mean": 'SRC_STATS["auto__string"][0]["mean__frequency"]',
                    "sd": 'SRC_STATS["auto__string"][0]["stddev__frequency"]',
                },
            )
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertSetEqual(
                set(gc.config["src-stats"][0].keys()), {"comments", "name", "query"}
            )
            self.assertEqual(gc.config["src-stats"][0]["name"], f"auto__string")
            self.assertEqual(
                gc.config["src-stats"][0]["query"],
                "SELECT AVG(frequency) AS mean__frequency, STDDEV(frequency) AS stddev__frequency FROM string",
            )

    def test_aggregate_queries_merge(self) -> None:
        """
        Test that we can set a generator that requires select aggregate clauses
        and keep an old one, resulting in a merged query.
        """
        config = {
            "tables": {
                "string": {
                    "row_generators": [
                        {
                            "name": "dist_gen.normal",
                            "columns_assigned": ["frequency"],
                            "kwargs": {
                                "mean": 'SRC_STATS["auto__string"]["results"][0]["mean__frequency"]',
                                "sd": 'SRC_STATS["auto__string"]["results"][0]["stddev__frequency"]',
                            },
                        }
                    ]
                }
            },
            "src-stats": [
                {
                    "name": "auto__string",
                    "query": "SELECT AVG(frequency) AS mean__frequency, STDDEV(frequency) AS stddev__frequency FROM string",
                }
            ],
        }
        with self._get_cmd(copy.deepcopy(config)) as gc:
            COLUMN = "position"
            GENERATOR = "dist_gen.uniform_ms"
            gc.do_next(f"string.{COLUMN}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[f"{GENERATOR}"][0]))
            gc.do_quit("")
            row_gens: list[dict[str, any]] = gc.config["tables"]["string"][
                "row_generators"
            ]
            self.assertEqual(len(row_gens), 2)
            if row_gens[0]["name"] == GENERATOR:
                row_gen0 = row_gens[0]
                row_gen1 = row_gens[1]
            else:
                row_gen0 = row_gens[1]
                row_gen1 = row_gens[0]
            self.assertEqual(row_gen0["name"], GENERATOR)
            self.assertEqual(row_gen1["name"], "dist_gen.normal")
            self.assertListEqual(row_gen0["columns_assigned"], [COLUMN])
            self.assertDictEqual(
                row_gen0["kwargs"],
                {
                    "mean": f'SRC_STATS["auto__string"]["results"][0]["mean__{COLUMN}"]',
                    "sd": f'SRC_STATS["auto__string"]["results"][0]["stddev__{COLUMN}"]',
                },
            )
            self.assertListEqual(row_gen1["columns_assigned"], ["frequency"])
            self.assertDictEqual(
                row_gen1["kwargs"],
                {
                    "mean": 'SRC_STATS["auto__string"]["results"][0]["mean__frequency"]',
                    "sd": 'SRC_STATS["auto__string"]["results"][0]["stddev__frequency"]',
                },
            )
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertEqual(gc.config["src-stats"][0]["name"], "auto__string")
            select_match = re.match(
                r"SELECT (.*) FROM string", gc.config["src-stats"][0]["query"]
            )
            self.assertIsNotNone(
                select_match, "src_stats[0].query is not an aggregate select"
            )
            self.assertSetEqual(
                set(select_match.group(1).split(", ")),
                {
                    "AVG(frequency) AS mean__frequency",
                    "STDDEV(frequency) AS stddev__frequency",
                    f"AVG({COLUMN}) AS mean__{COLUMN}",
                    f"STDDEV({COLUMN}) AS stddev__{COLUMN}",
                },
            )

    def test_next_completion(self) -> None:
        """Test tab completion for the next command."""
        with self._get_cmd({}) as gc:
            self.assertSetEqual(
                set(gc.complete_next("m", "next m", 5, 6)),
                {"manufacturer", "model"},
            )
            self.assertSetEqual(
                set(gc.complete_next("model", "next model", 5, 10)),
                {"model", "model."},
            )
            self.assertSetEqual(
                set(gc.complete_next("string.", "next string.", 5, 11)),
                {"string.id", "string.model_id", "string.position", "string.frequency"},
            )
            self.assertSetEqual(
                set(gc.complete_next("string.p", "next string.p", 5, 12)),
                {"string.position"},
            )
            self.assertListEqual(
                gc.complete_next("string.q", "next string.q", 5, 12), []
            )
            self.assertListEqual(gc.complete_next("ww", "next ww", 5, 7), [])

    def test_compare_reports_privacy(self) -> None:
        """
        Test that compare reports whether the current table is primary private,
        secondary private or not private.
        """
        config = {
            "tables": {
                "model": {
                    "primary_private": True,
                }
            },
        }
        with self._get_cmd(config) as gc:
            gc.do_next("manufacturer")
            gc.reset()
            gc.do_compare("")
            (text, args, _kwargs) = gc.messages[0]
            self.assertEqual(text, gc.NOT_PRIVATE_TEXT)
            gc.do_next("model")
            gc.reset()
            gc.do_compare("")
            (text, args, _kwargs) = gc.messages[0]
            self.assertEqual(text, gc.PRIMARY_PRIVATE_TEXT)
            gc.do_next("string")
            gc.reset()
            gc.do_compare("")
            (text, args, _kwargs) = gc.messages[0]
            self.assertEqual(text, gc.SECONDARY_PRIVATE_TEXT)
            self.assertSequenceEqual(args, [["model"]])

    def test_existing_configuration_remains(self) -> None:
        """
        Test setting a generator does not remove other information.
        """
        config = {
            "tables": {
                "string": {
                    "primary_private": True,
                }
            },
            "src-stats": [
                {
                    "name": "kraken",
                    "query": "SELECT MAX(frequency) AS max_frequency FROM string",
                }
            ],
        }
        with self._get_cmd(config) as gc:
            COLUMN = "position"
            GENERATOR = "dist_gen.uniform_ms"
            gc.do_next(f"string.{COLUMN}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[f"{GENERATOR}"][0]))
            gc.do_quit("")
            src_stats = {stat["name"]: stat["query"] for stat in gc.config["src-stats"]}
            self.assertEqual(src_stats["kraken"], config["src-stats"][0]["query"])
            self.assertTrue(gc.config["tables"]["string"]["primary_private"])

    def test_empty_tables_are_not_configured(self) -> None:
        """Test that tables marked as empty are not configured."""
        config = {
            "tables": {
                "string": {
                    "num_rows_per_pass": 0,
                }
            },
        }
        with self._get_cmd(copy.deepcopy(config)) as gc:
            gc.do_tables("")
            table_names = {m[1][0] for m in gc.messages}
            self.assertIn("model", table_names)
            self.assertNotIn("string", table_names)


class GeneratorsOutputTests(GeneratesDBTestCase):
    """Testing choice generation."""

    dump_file_path = "choice.sql"
    database_name = "numbers"
    schema_name = "public"

    def _get_cmd(self, config) -> TestGeneratorCmd:
        return TestGeneratorCmd(self.dsn, self.schema_name, self.metadata, config)

    def test_create_with_sampled_choice(self) -> None:
        """Test that suppression works for choice and zipf_choice."""
        table_name = "number_table"
        with self._get_cmd({}) as gc:
            gc.do_next("number_table.one")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            self.assertIn("dist_gen.choice", proposals)
            self.assertIn("dist_gen.zipf_choice", proposals)
            self.assertIn("dist_gen.choice [sampled]", proposals)
            self.assertIn("dist_gen.zipf_choice [sampled]", proposals)
            self.assertIn("dist_gen.choice [sampled and suppressed]", proposals)
            self.assertIn("dist_gen.zipf_choice [sampled and suppressed]", proposals)
            gc.do_set(str(proposals["dist_gen.choice [sampled and suppressed]"][0]))
            gc.do_next("number_table.two")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            self.assertIn("dist_gen.choice", proposals)
            self.assertIn("dist_gen.zipf_choice", proposals)
            self.assertIn("dist_gen.choice [sampled]", proposals)
            self.assertIn("dist_gen.zipf_choice [sampled]", proposals)
            self.assertIn("dist_gen.choice [sampled and suppressed]", proposals)
            self.assertIn("dist_gen.zipf_choice [sampled and suppressed]", proposals)
            gc.do_set(
                str(proposals["dist_gen.zipf_choice [sampled and suppressed]"][0])
            )
            gc.do_next("number_table.three")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            self.assertIn("dist_gen.choice", proposals)
            self.assertIn("dist_gen.zipf_choice", proposals)
            self.assertIn("dist_gen.choice [sampled]", proposals)
            self.assertIn("dist_gen.zipf_choice [sampled]", proposals)
            self.assertNotIn("dist_gen.choice [sampled and suppressed]", proposals)
            self.assertNotIn("dist_gen.zipf_choice [sampled and suppressed]", proposals)
            gc.do_set(str(proposals["dist_gen.choice [sampled]"][0]))
            gc.do_quit("")
            self.generate_data(gc.config, num_passes=200)
        with self.engine.connect() as conn:
            stmt = select(self.metadata.tables[table_name])
            rows = conn.execute(stmt).fetchall()
            ones = set()
            twos = set()
            threes = set()
            for row in rows:
                ones.add(row.one)
                twos.add(row.two)
                threes.add(row.three)
            # all generation possibilities should be present
            self.assertSetEqual(ones, {1, 4})
            self.assertSetEqual(twos, {2, 3})
            self.assertSetEqual(threes, {1, 2, 3, 4, 5})

    def test_create_with_choice(self) -> None:
        """Smoke test normal choice works."""
        table_name = "number_table"
        with self._get_cmd({}) as gc:
            gc.do_next("number_table.one")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals["dist_gen.choice"][0]))
            gc.do_next("number_table.two")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals["dist_gen.zipf_choice"][0]))
            gc.do_quit("")
            self.generate_data(gc.config, num_passes=200)
        with self.engine.connect() as conn:
            stmt = select(self.metadata.tables[table_name])
            rows = conn.execute(stmt).fetchall()
            ones = set()
            twos = set()
            for row in rows:
                ones.add(row.one)
                twos.add(row.two)
            # all generation possibilities should be present
            self.assertSetEqual(ones, {1, 2, 3, 4, 5})
            self.assertSetEqual(twos, {1, 2, 3, 4, 5})

    def test_create_with_weighted_choice(self) -> None:
        """Smoke test weighted choice."""
        table_name = "number_table"
        with self._get_cmd({}) as gc:
            gc.do_next("number_table.one")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            self.assertIn("dist_gen.weighted_choice", proposals)
            self.assertIn("dist_gen.weighted_choice [sampled]", proposals)
            self.assertIn(
                "dist_gen.weighted_choice [sampled and suppressed]", proposals
            )
            prop = proposals["dist_gen.weighted_choice [sampled and suppressed]"]
            self.assertSubset(set(prop[2]), {"1", "4"})
            gc.reset()
            gc.do_compare(str(prop[0]))
            col_heading = (
                f"{prop[0]}. dist_gen.weighted_choice [sampled and suppressed]"
            )
            self.assertIn(col_heading, set(gc.columns.keys()))
            self.assertSubset(set(gc.columns[col_heading]), {1, 4})
            gc.do_set(str(prop[0]))
            gc.do_next("number_table.two")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            self.assertIn("dist_gen.weighted_choice", proposals)
            self.assertIn("dist_gen.weighted_choice [sampled]", proposals)
            self.assertIn(
                "dist_gen.weighted_choice [sampled and suppressed]", proposals
            )
            prop = proposals["dist_gen.weighted_choice"]
            self.assertSubset(set(prop[2]), {"1", "2", "3", "4", "5"})
            gc.reset()
            gc.do_compare(str(prop[0]))
            col_heading = f"{prop[0]}. dist_gen.weighted_choice"
            self.assertIn(col_heading, set(gc.columns.keys()))
            self.assertSubset(set(gc.columns[col_heading]), {1, 2, 3, 4, 5})
            gc.do_set(str(prop[0]))
            gc.do_next("number_table.three")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            self.assertIn("dist_gen.weighted_choice", proposals)
            self.assertIn("dist_gen.weighted_choice [sampled]", proposals)
            self.assertNotIn(
                "dist_gen.weighted_choice [sampled and suppressed]", proposals
            )
            prop = proposals["dist_gen.weighted_choice [sampled]"]
            self.assertSubset(set(prop[2]), {"1", "2", "3", "4", "5"})
            gc.do_compare(str(prop[0]))
            col_heading = f"{prop[0]}. dist_gen.weighted_choice [sampled]"
            self.assertIn(col_heading, set(gc.columns.keys()))
            self.assertSubset(set(gc.columns[col_heading]), {1, 2, 3, 4, 5})
            gc.do_set(str(prop[0]))
            gc.do_quit("")
            self.generate_data(gc.config, num_passes=200)
        with self.engine.connect() as conn:
            stmt = select(self.metadata.tables[table_name])
            rows = conn.execute(stmt).fetchall()
            ones = set()
            twos = set()
            threes = set()
            for row in rows:
                ones.add(row.one)
                twos.add(row.two)
                threes.add(row.three)
            # all generation possibilities should be present
            self.assertSetEqual(ones, {1, 4})
            self.assertSetEqual(twos, {1, 2, 3, 4, 5})
            self.assertSetEqual(threes, {1, 2, 3, 4, 5})


class TestMissingnessCmd(MissingnessCmd, TestDbCmdMixin):
    """MissingnessCmd but mocked"""


class ConfigureMissingnessTests(RequiresDBTestCase):
    """Testing configure-missing."""

    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    def _get_cmd(self, config) -> TestMissingnessCmd:
        return TestMissingnessCmd(self.dsn, self.schema_name, self.metadata, config)

    def test_set_missingness_to_sampled(self) -> None:
        """Test that we can set one table to sampled missingness."""
        with self._get_cmd({}) as mc:
            TABLE = "signature_model"
            mc.do_next(TABLE)
            mc.do_counts("")
            self.assertListEqual(
                mc.messages, [(MissingnessCmd.ROW_COUNT_MSG, (6,), {})]
            )
            self.assertListEqual(mc.rows, [["player_id", 3], ["based_on", 2]])
            mc.do_sampled("")
            mc.do_quit("")
            self.assertDictEqual(
                mc.config,
                {
                    "tables": {
                        TABLE: {
                            "missingness_generators": [
                                {
                                    "columns": ["player_id", "based_on"],
                                    "kwargs": {
                                        "patterns": 'SRC_STATS["missing_auto__signature_model__0"]'
                                    },
                                    "name": "column_presence.sampled",
                                }
                            ]
                        }
                    },
                    "src-stats": [
                        {
                            "name": "missing_auto__signature_model__0",
                            "query": (
                                "SELECT COUNT(*) AS row_count, player_id__is_null, based_on__is_null FROM"
                                " (SELECT player_id IS NULL AS player_id__is_null, based_on IS NULL AS based_on__is_null FROM"
                                " signature_model ORDER BY RANDOM() LIMIT 1000) AS __t GROUP BY player_id__is_null, based_on__is_null"
                            ),
                        }
                    ],
                },
            )


class ConfigureMissingnessTests(GeneratesDBTestCase):
    """Testing configure-missing with generation."""

    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    def _get_cmd(self, config) -> TestMissingnessCmd:
        return TestMissingnessCmd(self.dsn, self.schema_name, self.metadata, config)

    def test_create_with_missingness(self) -> None:
        """Test that we can sample real missingness and reproduce it."""
        random.seed(45)
        # Configure the missingness
        table_name = "signature_model"
        with self._get_cmd({}) as mc:
            mc.do_next(table_name)
            mc.do_sampled("")
            mc.do_quit("")
            config = mc.config
            self.generate_data(config, num_passes=100)
        # Test that each missingness pattern is present in the database
        with self.engine.connect() as conn:
            stmt = select(self.metadata.tables[table_name])
            rows = conn.execute(stmt).mappings().fetchall()
            patterns: set[int] = set()
            for row in rows:
                p = 0 if row["player_id"] is None else 1
                b = 0 if row["based_on"] is None else 2
                patterns.add(p + b)
            # all pattern possibilities should be present
            self.assertSetEqual(patterns, {0, 1, 2, 3})


class GeneratorTests(GeneratesDBTestCase):
    """Testing configure-generators with generation."""

    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    def _get_cmd(self, config) -> TestGeneratorCmd:
        return TestGeneratorCmd(self.dsn, self.schema_name, self.metadata, config)

    def test_set_null(self) -> None:
        """Test that we can sample real missingness and reproduce it."""
        with self._get_cmd({}) as gc:
            gc.do_next("string.position")
            gc.do_set("dist_gen.constant")
            self.assertListEqual(gc.messages, [])
            gc.reset()
            gc.do_next("string.frequency")
            gc.do_set("dist_gen.constant")
            self.assertListEqual(gc.messages, [])
            gc.reset()
            gc.do_next("signature_model.name")
            gc.do_set("dist_gen.constant")
            self.assertListEqual(gc.messages, [])
            gc.reset()
            gc.do_next("signature_model.based_on")
            gc.do_set("dist_gen.constant")
            # we have got to the end of the columns, but shouldn't have any errors
            self.assertListEqual(
                gc.messages, [(GeneratorCmd.INFO_NO_MORE_TABLES, (), {})]
            )
            gc.reset()
            gc.do_quit("")
            config = gc.config
            self.generate_data(config, num_passes=3)
        # Test that each missingness pattern is present in the database
        with self.engine.connect() as conn:
            stmt = select(self.metadata.tables["string"].c["position", "frequency"])
            rows = conn.execute(stmt).fetchall()
            count = 0
            for row in rows:
                count += 1
                self.assertEqual(row.position, 0)
                self.assertEqual(row.frequency, 0.0)
            self.assertEqual(count, 3)
            stmt = select(self.metadata.tables["signature_model"].c["name", "based_on"])
            rows = conn.execute(stmt).fetchall()
            count = 0
            for row in rows:
                count += 1
                self.assertEqual(row.name, "")
                self.assertIsNone(row.based_on)
            self.assertEqual(count, 3)

    def test_dist_gen_sampled_produces_ordered_src_stats(self) -> None:
        """Tests that choosing a sampled choice generator produces ordered src stats"""
        with self._get_cmd({}) as gc:
            gc.do_next("signature_model.player_id")
            gc.do_set("dist_gen.zipf_choice [sampled]")
            gc.do_next("signature_model.based_on")
            gc.do_set("dist_gen.zipf_choice [sampled]")
            gc.do_quit("")
            config = gc.config
            self.set_configuration(config)
            src_stats = self.get_src_stats(config)
        player_ids = [
            s["value"] for s in src_stats["auto__signature_model__player_id"]["results"]
        ]
        self.assertListEqual(player_ids, [2, 3, 1])
        based_ons = [
            s["value"] for s in src_stats["auto__signature_model__based_on"]["results"]
        ]
        self.assertListEqual(based_ons, [1, 3, 2])

    def assertAreTruncatedTo(self, xs, length) -> None:
        maxlen = 0
        for x in xs:
            newlen = len(x.strip("'\""))
            self.assertLessEqual(newlen, length)
            maxlen = max(maxlen, newlen)
        self.assertEqual(maxlen, length)

    def test_varchar_ns_are_truncated(self) -> None:
        """Tests that mimesis generators for VARCHAR(N) truncate to N characters"""
        GENERATOR = "generic.text.quote"
        TABLE = "signature_model"
        COLUMN = "name"
        with self._get_cmd({}) as gc:
            gc.do_next(f"{TABLE}.{COLUMN}")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            quotes = [k for k in proposals.keys() if k.startswith(GENERATOR)]
            self.assertEqual(len(quotes), 1)
            prop = proposals[quotes[0]]
            self.assertAreTruncatedTo(prop[2], 20)
            gc.reset()
            gc.do_compare(str(prop[0]))
            col_heading = f"{prop[0]}. {quotes[0]}"
            gc.do_set(str(prop[0]))
            self.assertIn(col_heading, gc.columns)
            self.assertAreTruncatedTo(gc.columns[col_heading], 20)
            gc.do_quit("")
            config = gc.config
            self.generate_data(config, num_passes=15)
        with self.engine.connect() as conn:
            stmt = select(self.metadata.tables[TABLE].c[COLUMN])
            rows = conn.execute(stmt).scalars().fetchall()
            self.assertAreTruncatedTo(rows, 20)


@dataclass
class Stat:
    n: int = 0
    x: float = 0
    x2: float = 0

    def add(self, x: float) -> None:
        self.n += 1
        self.x += x
        self.x2 += x * x

    def count(self) -> int:
        return self.n

    def x_mean(self) -> float:
        return self.x / self.n

    def x_var(self) -> float:
        x = self.x
        return (self.x2 - x * x / self.n) / (self.n - 1)


@dataclass
class Correlation(Stat):
    y: float = 0
    y2: float = 0
    xy: float = 0

    def add(self, x: float, y: float) -> None:
        self.n += 1
        self.x += x
        self.x2 += x * x
        self.y += y
        self.y2 += y * y
        self.xy += x * y

    def y_mean(self) -> float:
        return self.y / self.n

    def y_var(self) -> float:
        y = self.y
        return (self.y2 - y * y / self.n) / (self.n - 1)

    def covar(self) -> float:
        return (self.xy - self.x * self.y / self.n) / (self.n - 1)


class NullPartitionedTests(GeneratesDBTestCase):
    """Testing null-partitioned grouped multivariate generation."""

    dump_file_path = "eav.sql"
    database_name = "eav"
    schema_name = "public"

    def setUp(self) -> None:
        super().setUp()
        NullPartitionedNormalGeneratorFactory.SAMPLE_COUNT = 8
        NullPartitionedNormalGeneratorFactory.SUPPRESS_COUNT = 2

    def _get_cmd(self, config) -> TestGeneratorCmd:
        return TestGeneratorCmd(self.dsn, self.schema_name, self.metadata, config)

    def test_create_with_null_partitioned_grouped_multivariate(self) -> None:
        """Test EAV for all columns."""
        table_name = "measurement"
        generate_count = 800
        with self._get_cmd({}) as gc:
            gc.do_next("measurement.type")
            gc.do_merge("first_value")
            gc.do_merge("second_value")
            gc.do_merge("third_value")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            self.assertIn("null-partitioned grouped_multivariate_lognormal", proposals)
            dist_to_choose = "null-partitioned grouped_multivariate_normal"
            self.assertIn(dist_to_choose, proposals)
            prop = proposals[dist_to_choose]
            gc.reset()
            gc.do_compare(str(prop[0]))
            col_heading = f"{prop[0]}. {dist_to_choose}"
            self.assertIn(col_heading, set(gc.columns.keys()))
            gc.do_set(str(prop[0]))
            gc.reset()
            gc.do_quit("")
            self.set_configuration(gc.config)
            self.get_src_stats(gc.config)
            self.create_generators(gc.config)
            self.remove_data(gc.config)
            # let's add a vocab table without messing around with files
            table = self.metadata.tables["measurement_type"]
            with self.engine.connect() as conn:
                conn.execute(insert(table).values({"id": 1, "name": "agreement"}))
                conn.execute(insert(table).values({"id": 2, "name": "acceleration"}))
                conn.execute(insert(table).values({"id": 3, "name": "velocity"}))
                conn.execute(insert(table).values({"id": 4, "name": "position"}))
                conn.execute(insert(table).values({"id": 5, "name": "matter"}))
                conn.commit()
            self.create_data(gc.config, num_passes=generate_count)
        with self.engine.connect() as conn:
            stmt = select(self.metadata.tables[table_name])
            rows = conn.execute(stmt).fetchall()
            one_count = 0
            one_yes_count = 0
            two = Correlation()
            three = Correlation()
            four = Correlation()
            fish = Stat()
            fowl = Stat()
            for row in rows:
                if row.type == 1:
                    # yes or no
                    self.assertIsNone(row.first_value)
                    self.assertIsNone(row.second_value)
                    self.assertIn(row.third_value, {"yes", "no"})
                    one_count += 1
                    if row.third_value == "yes":
                        one_yes_count += 1
                elif row.type == 2:
                    # positive correlation around 1.4, 1.8
                    self.assertIsNotNone(row.first_value)
                    self.assertIsNotNone(row.second_value)
                    self.assertIsNone(row.third_value)
                    two.add(row.first_value, row.second_value)
                elif row.type == 3:
                    # negative correlation around 11.8, 12.1
                    self.assertIsNotNone(row.first_value)
                    self.assertIsNotNone(row.second_value)
                    self.assertIsNone(row.third_value)
                    three.add(row.first_value, row.second_value)
                elif row.type == 4:
                    # positive correlation around 21.4, 23.4
                    self.assertIsNotNone(row.first_value)
                    self.assertIsNotNone(row.second_value)
                    self.assertIsNone(row.third_value)
                    four.add(row.first_value, row.second_value)
                elif row.type == 5:
                    self.assertIn(row.third_value, {"fish", "fowl"})
                    self.assertIsNotNone(row.first_value)
                    self.assertIsNone(row.second_value)
                    if row.third_value == "fish":
                        # mean 8.1 and sd 0.755
                        fish.add(row.first_value)
                    else:
                        # mean 11.2 and sd 1.114
                        fowl.add(row.first_value)
            # type 1
            self.assertAlmostEqual(
                one_count, generate_count * 5 / 20, delta=generate_count * 0.4
            )
            # about 40% are yes
            self.assertAlmostEqual(
                one_yes_count / one_count, 0.4, delta=generate_count * 0.4
            )
            # type 2
            self.assertAlmostEqual(
                two.count(), generate_count * 3 / 20, delta=generate_count * 0.5
            )
            self.assertAlmostEqual(two.x_mean(), 1.4, delta=0.6)
            self.assertAlmostEqual(two.x_var(), 0.21, delta=0.4)
            self.assertAlmostEqual(two.y_mean(), 1.8, delta=0.8)
            self.assertAlmostEqual(two.y_var(), 0.07, delta=0.1)
            self.assertAlmostEqual(two.covar(), 0.5, delta=0.5)
            # type 3
            self.assertAlmostEqual(
                three.count(), generate_count * 3 / 20, delta=generate_count * 0.2
            )
            self.assertAlmostEqual(two.covar(), -0.5, delta=0.5)
            # type 4
            self.assertAlmostEqual(
                four.count(), generate_count * 3 / 20, delta=generate_count * 0.2
            )
            self.assertAlmostEqual(two.covar(), 0.5, delta=0.5)
            # type 5/fish
            self.assertAlmostEqual(
                fish.count(), generate_count * 3 / 20, delta=generate_count * 0.2
            )
            self.assertAlmostEqual(fish.x_mean(), 8.1, delta=3.0)
            self.assertAlmostEqual(fish.x_var(), 0.57, delta=0.8)
            # type 5/fowl
            self.assertAlmostEqual(
                fowl.count(), generate_count * 3 / 20, delta=generate_count * 0.2
            )
            self.assertAlmostEqual(fish.x_mean(), 11.2, delta=8.0)
            self.assertAlmostEqual(fish.x_var(), 1.24, delta=1.5)

    def test_create_with_null_partitioned_grouped_sampled_and_suppressed(self) -> None:
        """Test EAV for all columns with sampled and suppressed generation."""
        table_name = "measurement"
        table2_name = "observation"
        generate_count = 800
        with self._get_cmd({}) as gc:
            gc.do_next("measurement.type")
            gc.do_merge("first_value")
            gc.do_merge("second_value")
            gc.do_merge("third_value")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            self.assertIn("null-partitioned grouped_multivariate_lognormal", proposals)
            self.assertIn("null-partitioned grouped_multivariate_normal", proposals)
            self.assertIn(
                "null-partitioned grouped_multivariate_lognormal [sampled and suppressed]",
                proposals,
            )
            dist_to_choose = (
                "null-partitioned grouped_multivariate_normal [sampled and suppressed]"
            )
            self.assertIn(dist_to_choose, proposals)
            prop = proposals[dist_to_choose]
            gc.reset()
            gc.do_compare(str(prop[0]))
            col_heading = f"{prop[0]}. {dist_to_choose}"
            self.assertIn(col_heading, set(gc.columns.keys()))
            gc.do_set(str(prop[0]))
            gc.reset()
            gc.do_next("observation.type")
            gc.do_merge("first_value")
            gc.do_merge("second_value")
            gc.do_merge("third_value")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            dist_to_choose = (
                "null-partitioned grouped_multivariate_normal [sampled and suppressed]"
            )
            prop = proposals[dist_to_choose]
            gc.do_set(str(prop[0]))
            gc.do_quit("")
            self.set_configuration(gc.config)
            self.get_src_stats(gc.config)
            self.create_generators(gc.config)
            self.remove_data(gc.config)
            # let's add a vocab table without messing around with files
            table = self.metadata.tables["measurement_type"]
            with self.engine.connect() as conn:
                conn.execute(insert(table).values({"id": 1, "name": "agreement"}))
                conn.execute(insert(table).values({"id": 2, "name": "acceleration"}))
                conn.execute(insert(table).values({"id": 3, "name": "velocity"}))
                conn.execute(insert(table).values({"id": 4, "name": "position"}))
                conn.execute(insert(table).values({"id": 5, "name": "matter"}))
                conn.commit()
            self.create_data(gc.config, num_passes=generate_count)
        with self.engine.connect() as conn:
            stmt = select(self.metadata.tables[table_name])
            rows = conn.execute(stmt).fetchall()
            one_count = 0
            one_yes_count = 0
            fish = Stat()
            fowl = Stat()
            types: set[int] = set()
            for row in rows:
                types.add(row.type)
                if row.type == 1:
                    # yes or no
                    self.assertIsNone(row.first_value)
                    self.assertIsNone(row.second_value)
                    self.assertIn(row.third_value, {"yes", "no"})
                    if row.third_value == "yes":
                        one_yes_count += 1
                    one_count += 1
                elif row.type == 5:
                    self.assertIn(row.third_value, {"fish", "fowl"})
                    self.assertIsNotNone(row.first_value)
                    self.assertIsNone(row.second_value)
                    if row.third_value == "fish":
                        # mean 8.1 and sd 0.755
                        fish.add(row.first_value)
                    else:
                        # mean 11.2 and sd 1.114
                        fowl.add(row.first_value)
            self.assertSubset(types, {1, 2, 3, 4, 5})
            self.assertEqual(len(types), 4)
            self.assertSubset({1, 5}, types)
            # type 1
            self.assertAlmostEqual(
                one_count, generate_count * 5 / 11, delta=generate_count * 0.4
            )
            # about 40% are yes
            self.assertAlmostEqual(
                one_yes_count / one_count, 0.4, delta=generate_count * 0.4
            )
            # type 5/fish
            self.assertAlmostEqual(
                fish.count(), generate_count * 3 / 11, delta=generate_count * 0.2
            )
            self.assertAlmostEqual(fish.x_mean(), 8.1, delta=3.0)
            self.assertAlmostEqual(fish.x_var(), 0.57, delta=0.8)
            # type 5/fowl
            self.assertAlmostEqual(
                fowl.count(), generate_count * 3 / 11, delta=generate_count * 0.2
            )
            self.assertAlmostEqual(fish.x_mean(), 11.2, delta=8.0)
            self.assertAlmostEqual(fish.x_var(), 1.24, delta=1.5)
            stmt = select(self.metadata.tables[table2_name])
            rows = conn.execute(stmt).fetchall()
            firsts = Stat()
            for row in rows:
                types.add(row.type)
                self.assertEqual(row.type, 1)
                self.assertIsNotNone(row.first_value)
                self.assertIsNone(row.second_value)
                self.assertIn(row.third_value, {"ham", "eggs"})
                firsts.add(row.first_value)
            self.assertEqual(firsts.count(), 800)
            self.assertAlmostEqual(firsts.x_mean(), 1.3, delta=generate_count * 0.3)


class NonInteractiveTests(RequiresDBTestCase):
    """
    Test the --spec SPEC_FILE option of configure-generators
    """

    dump_file_path = "eav.sql"
    database_name = "eav"
    schema_name = "public"

    @patch("datafaker.interactive.Path")
    @patch(
        "datafaker.interactive.csv.reader",
        return_value=iter(
            [
                ["observation", "type", "dist_gen.weighted_choice [sampled]"],
                [
                    "observation",
                    "first_value",
                    "dist_gen.weighted_choice",
                    "dist_gen.constant",
                ],
                [
                    "observation",
                    "second_value",
                    "dist_gen.weighted_choice",
                    "dist_gen.weighted_choice [sampled]",
                    "dist_gen.constant",
                ],
                ["observation", "third_value", "dist_gen.weighted_choice"],
            ]
        ),
    )
    def test_non_interactive_configure_generators(
        self, _mock_csv_reader: MagicMock, _mock_path: MagicMock
    ) -> None:
        """
        test that we can set generators from a CSV file
        """
        config = {}
        spec_csv = Mock(return_value="mock spec.csv file")
        update_config_generators(
            self.dsn, self.schema_name, self.metadata, config, spec_csv
        )
        row_gens = {
            f"{table}{sorted(rg['columns_assigned'])}": rg["name"]
            for table, tables in config.get("tables", {}).items()
            for rg in tables.get("row_generators", [])
        }
        self.assertEqual(row_gens["observation['type']"], "dist_gen.weighted_choice")
        self.assertEqual(
            row_gens["observation['first_value']"], "dist_gen.weighted_choice"
        )
        self.assertEqual(row_gens["observation['second_value']"], "dist_gen.constant")
        self.assertEqual(
            row_gens["observation['third_value']"], "dist_gen.weighted_choice"
        )

    @patch("datafaker.interactive.Path")
    @patch(
        "datafaker.interactive.csv.reader",
        return_value=iter(
            [
                [
                    "observation",
                    "type first_value second_value third_value",
                    "null-partitioned grouped_multivariate_lognormal",
                ],
            ]
        ),
    )
    def test_non_interactive_configure_null_partitioned(
        self, _mock_csv_reader: MagicMock, _mock_path: MagicMock
    ) -> None:
        """
        test that we can set multi-column generators from a CSV file
        """
        config = {}
        spec_csv = Mock(return_value="mock spec.csv file")
        update_config_generators(
            self.dsn, self.schema_name, self.metadata, config, spec_csv
        )
        row_gens = {
            f"{table}{sorted(rg['columns_assigned'])}": rg
            for table, tables in config.get("tables", {}).items()
            for rg in tables.get("row_generators", [])
        }
        self.assertEqual(
            row_gens[
                "observation['first_value', 'second_value', 'third_value', 'type']"
            ]["name"],
            "dist_gen.alternatives",
        )
        self.assertEqual(
            row_gens[
                "observation['first_value', 'second_value', 'third_value', 'type']"
            ]["kwargs"]["alternative_configs"][0]["name"],
            '"with_constants_at"',
        )
        self.assertEqual(
            row_gens[
                "observation['first_value', 'second_value', 'third_value', 'type']"
            ]["kwargs"]["alternative_configs"][0]["params"]["subgen"],
            '"grouped_multivariate_lognormal"',
        )
