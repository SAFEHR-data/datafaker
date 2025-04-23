"""Tests for the base module."""
import copy
import re
from sqlalchemy import MetaData, select
from sqlalchemy.orm import declarative_base

from sqlsynthgen.interactive import DbCmd, TableCmd, GeneratorCmd
from tests.utils import RequiresDBTestCase


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
    """ TableCmd but mocked """


class ConfigureTablesTests(RequiresDBTestCase):
    """Testing configure-tables."""
    dump_file_path = "src.dump"
    database_name = "src"
    schema_name = "public"

    def test_table_name_prompts(self) -> None:
        """Test that the prompts follow the names of the tables."""
        metadata = MetaData()
        metadata.reflect(self.engine)
        config = {}
        with TestTableCmd(self.dsn, self.schema_name, metadata, config) as tc:
            table_names = list(metadata.tables.keys())
            for t in table_names:
                self.assertIn(t, tc.prompt)
                tc.do_next("")
            self.assertListEqual(tc.messages, [(TableCmd.ERROR_NO_MORE_TABLES, (), {})])
            tc.reset()
            for t in reversed(table_names):
                self.assertIn(t, tc.prompt)
                tc.do_previous("")
            self.assertListEqual(tc.messages, [(TableCmd.ERROR_ALREADY_AT_START, (), {})])
            tc.reset()
            bad_table_name = "notarealtable"
            tc.do_next(bad_table_name)
            self.assertListEqual(tc.messages, [(TableCmd.ERROR_NO_SUCH_TABLE, (bad_table_name,), {})])
            tc.reset()
            good_table_name = table_names[2]
            tc.do_next(good_table_name)
            self.assertListEqual(tc.messages, [])
            self.assertIn(good_table_name, tc.prompt)

    def test_column_display(self) -> None:
        """Test that we can see the names of the columns."""
        metadata = MetaData()
        metadata.reflect(self.engine)
        config = {}
        with TestTableCmd(self.dsn, self.schema_name, metadata, config) as tc:
            tc.do_next("unique_constraint_test")
            tc.do_columns("")
            self.assertListEqual(tc.column_items, [["id", "a", "b", "c"]])

    def test_configure_tables(self) -> None:
        """Test that we can change columns to ignore, vocab or normal."""
        metadata = MetaData()
        metadata.reflect(self.engine)
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
        with TestTableCmd(self.dsn, self.schema_name, metadata, config) as tc:
            tc.do_next("unique_constraint_test")
            tc.do_normal("")
            tc.do_next("person")
            tc.do_vocabulary("")
            tc.do_next("mitigation_type")
            tc.do_ignore("")
            tc.do_quit("")
            tables = tc.config["tables"]
            self.assertFalse(tables["unique_constraint_test"].get("vocabulary_table", False))
            self.assertFalse(tables["unique_constraint_test"].get("ignore", False))
            self.assertFalse(tables["no_pk_test"].get("vocabulary_table", False))
            self.assertTrue(tables["no_pk_test"].get("ignore", False))
            self.assertTrue(tables["person"].get("vocabulary_table", False))
            self.assertFalse(tables["person"].get("ignore", False))
            self.assertFalse(tables["mitigation_type"].get("vocabulary_table", False))
            self.assertTrue(tables["mitigation_type"].get("ignore", False))

    def test_print_data(self) -> None:
        """Test that we can print random rows from the table and random data from columns."""
        metadata = MetaData()
        metadata.reflect(self.engine)
        person_table = metadata.tables["person"]
        with self.engine.connect() as conn:
            person_rows = conn.execute(select(person_table)).mappings().fetchall()
            person_data = {
                row["person_id"]: row
                for row in person_rows
            }
            name_set = {row["name"] for row in person_rows}
        person_headings = ["person_id", "name", "research_opt_out", "stored_from"]
        config = {}
        with TestTableCmd(self.dsn, self.schema_name, metadata, config) as tc:
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
            self.assertEqual(set(tc.column_items[0]), set(filter(lambda n: 13 <= len(n), name_set)))
            tc.reset()
            tc.do_data(f"{to_get_count} name 16")
            self.assertEqual(len(tc.column_items), 1)
            self.assertEqual(set(tc.column_items[0]), set(filter(lambda n: 16 <= len(n), name_set)))

    def test_list_tables(self):
        """Test that we can list the tables"""
        metadata = MetaData()
        metadata.reflect(self.engine)
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
        with TestTableCmd(self.dsn, self.schema_name, metadata, config) as tc:
            tc.do_next("unique_constraint_test")
            tc.do_ignore("")
            tc.do_next("person")
            tc.do_vocabulary("")
            tc.reset()
            tc.do_list("")
            person_listed = False
            unique_constraint_test_listed = False
            no_pk_test_listed = False
            for (text, args, kwargs) in tc.messages:
                if args[2] == "person":
                    self.assertFalse(person_listed)
                    person_listed = True
                    self.assertEqual(args[0], " ")
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
                    self.assertEqual(args[0], " ")
                    self.assertEqual(args[1], "   ")
            self.assertTrue(person_listed)
            self.assertTrue(unique_constraint_test_listed)
            self.assertTrue(no_pk_test_listed)


class TestGeneratorCmd(GeneratorCmd, TestDbCmdMixin):
    """ TableCmd but mocked """
    def get_proposals(self) -> dict[str, tuple[int, str, str, list[str]]]:
        """
        Returns a dict of generator name to a tuple of (index, fit_string, [list,of,samples])"""
        return {
            kw["name"]: (kw["index"], kw["fit"], kw["sample"].split(", "))
            for (s, _, kw) in self.messages
            if s == self.PROPOSE_GENERATOR_SAMPLE_TEXT
        }


class ConfigureGeneratorsTests(RequiresDBTestCase):
    """ Testing configure-generators. """
    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    def test_set_generator_mimesis(self):
        """ Test that we can set one generator to a mimesis generator. """
        metadata = MetaData()
        metadata.reflect(self.engine)
        with TestGeneratorCmd(self.dsn, self.schema_name, metadata, {}) as gc:
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

    def test_set_generator_distribution(self):
        """ Test that we can set one generator to gaussian. """
        metadata = MetaData()
        metadata.reflect(self.engine)
        with TestGeneratorCmd(self.dsn, self.schema_name, metadata, {}) as gc:
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
            self.assertDictEqual(row_gen["kwargs"], {
                "mean": f'SRC_STATS["auto__{TABLE}"]["mean__{COLUMN}"]',
                "sd": f'SRC_STATS["auto__{TABLE}"]["stddev__{COLUMN}"]',
            })
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertDictEqual(gc.config["src-stats"][0], {
                "name": f"auto__{TABLE}",
                "query": f"SELECT AVG({COLUMN}) AS mean__{COLUMN}, STDDEV({COLUMN}) AS stddev__{COLUMN} FROM {TABLE}",
            })

    def test_set_generator_choice(self):
        """ Test that we can set one generator to uniform choice. """
        metadata = MetaData()
        metadata.reflect(self.engine)
        with TestGeneratorCmd(self.dsn, self.schema_name, metadata, {}) as gc:
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
            self.assertDictEqual(row_gen["kwargs"], {
                "a": f'SRC_STATS["auto__{TABLE}__{COLUMN}"]["value"]',
            })
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertDictEqual(gc.config["src-stats"][0], {
                "name": f"auto__{TABLE}__{COLUMN}",
                "query": f"SELECT {COLUMN} AS value FROM {TABLE} GROUP BY value ORDER BY COUNT({COLUMN}) DESC",
            })

    def test_old_generators_remain(self):
        """ Test that we can set one generator and keep an old one. """
        metadata = MetaData()
        metadata.reflect(self.engine)
        config = {
            "tables": {
                "string": {
                    "row_generators": [{
                        "name": "dist_gen.normal",
                        "columns_assigned": ["frequency"],
                        "kwargs": {
                            "mean": 'SRC_STATS["auto__string"]["mean__frequency"]',
                            "sd": 'SRC_STATS["auto__string"]["stddev__frequency"]',
                        },
                    }]
                }
            },
            "src-stats": [{
                "name": "auto__string",
                "query": 'SELECT AVG(frequency) AS mean__frequency, STDDEV(frequency) AS stddev__frequency FROM string',
            }]
        }
        with TestGeneratorCmd(self.dsn, self.schema_name, metadata, copy.deepcopy(config)) as gc:
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
            self.assertDictEqual(row_gen["kwargs"], {
                "mean": 'SRC_STATS["auto__string"]["mean__frequency"]',
                "sd": 'SRC_STATS["auto__string"]["stddev__frequency"]',
            })
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertDictEqual(gc.config["src-stats"][0], {
                "name": "auto__string",
                "query": "SELECT AVG(frequency) AS mean__frequency, STDDEV(frequency) AS stddev__frequency FROM string",
            })
    
    def test_aggregate_queries_merge(self):
        """
        Test that we can set a generator that requires select aggregate clauses
        and keep an old one, resulting in a merged query.
        """
        metadata = MetaData()
        metadata.reflect(self.engine)
        config = {
            "tables": {
                "string": {
                    "row_generators": [{
                        "name": "dist_gen.normal",
                        "columns_assigned": ["frequency"],
                        "kwargs": {
                            "mean": 'SRC_STATS["auto__string"]["mean__frequency"]',
                            "sd": 'SRC_STATS["auto__string"]["stddev__frequency"]',
                        },
                    }]
                }
            },
            "src-stats": [{
                "name": "auto__string",
                "query": 'SELECT AVG(frequency) AS mean__frequency, STDDEV(frequency) AS stddev__frequency FROM string',
            }]
        }
        with TestGeneratorCmd(self.dsn, self.schema_name, metadata, copy.deepcopy(config)) as gc:
            COLUMN = "position"
            GENERATOR = "dist_gen.uniform_ms"
            gc.do_next(f"string.{COLUMN}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[f"{GENERATOR}"][0]))
            gc.do_quit("")
            row_gens: list[dict[str,any]] = gc.config["tables"]["string"]["row_generators"]
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
            self.assertDictEqual(row_gen0["kwargs"], {
                "mean": f'SRC_STATS["auto__string"]["mean__{COLUMN}"]',
                "sd": f'SRC_STATS["auto__string"]["stddev__{COLUMN}"]',
            })
            self.assertListEqual(row_gen1["columns_assigned"], ["frequency"])
            self.assertDictEqual(row_gen1["kwargs"], {
                "mean": 'SRC_STATS["auto__string"]["mean__frequency"]',
                "sd": 'SRC_STATS["auto__string"]["stddev__frequency"]',
            })
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertEqual(gc.config["src-stats"][0]["name"], "auto__string")
            select_match = re.match(r'SELECT (.*) FROM string', gc.config["src-stats"][0]["query"])
            self.assertIsNotNone(select_match, "src_stats[0].query is not an aggregate select")
            self.assertSetEqual(set(select_match.group(1).split(", ")), {
                "AVG(frequency) AS mean__frequency",
                "STDDEV(frequency) AS stddev__frequency",
                f"AVG({COLUMN}) AS mean__{COLUMN}",
                f"STDDEV({COLUMN}) AS stddev__{COLUMN}",
            })

    def test_next_completion(self):
        """ Test tab completion for the next command. """
        metadata = MetaData()
        metadata.reflect(self.engine)
        with TestGeneratorCmd(self.dsn, self.schema_name, metadata, {}) as gc:
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
            self.assertListEqual(gc.complete_next("string.q", "next string.q", 5, 12), [])
            self.assertListEqual(gc.complete_next("ww", "next ww", 5, 7), [])

    def test_compare_reports_privacy(self):
        """
        Test that compare reports whether the current table is primary private,
        secondary private or not private.
        """
        metadata = MetaData()
        metadata.reflect(self.engine)
        config = {
            "tables": {
                "model": {
                    "primary_private": True,
                }
            },
        }
        with TestGeneratorCmd(self.dsn, self.schema_name, metadata, copy.deepcopy(config)) as gc:
            gc.do_next("manufacturer")
            gc.reset()
            gc.do_compare("")
            (text, args, kwargs) = gc.messages[0]
            self.assertEqual(text, gc.NOT_PRIVATE_TEXT)
            gc.do_next("model")
            gc.reset()
            gc.do_compare("")
            (text, args, kwargs) = gc.messages[0]
            self.assertEqual(text, gc.PRIMARY_PRIVATE_TEXT)
            gc.do_next("string")
            gc.reset()
            gc.do_compare("")
            (text, args, kwargs) = gc.messages[0]
            self.assertEqual(text, gc.SECONDARY_PRIVATE_TEXT)
            self.assertSequenceEqual(args, [["model"]])
