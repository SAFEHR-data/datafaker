""" Tests for the configure-generators command. """
import copy
import re
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Iterable
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from sqlalchemy import Connection, MetaData, insert, select

from datafaker.generators import NullPartitionedNormalGeneratorFactory
from datafaker.generators.choice import ChoiceGeneratorFactory
from datafaker.interactive import update_config_generators
from datafaker.interactive.generators import GeneratorCmd
from tests.utils import GeneratesDBTestCase, RequiresDBTestCase, TestDbCmdMixin


class TestGeneratorCmd(GeneratorCmd, TestDbCmdMixin):
    """GeneratorCmd but mocked"""

    def get_proposals(self) -> dict[str, tuple[int, str, list[str]]]:
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

    def _get_cmd(self, config: MutableMapping[str, Any]) -> TestGeneratorCmd:
        """Get the command we are using for this test case."""
        return TestGeneratorCmd(self.dsn, self.schema_name, self.metadata, config)

    def test_null_configuration(self) -> None:
        """Test that the tables having null configuration does not break."""
        config = {
            "tables": None,
        }
        with self._get_cmd(config) as gc:
            table = "model"
            gc.do_next(f"{table}.name")
            gc.do_propose("")
            gc.do_compare("")
            gc.do_set("1")
            gc.do_quit("")
            self.assertEqual(len(gc.config["tables"][table]["row_generators"]), 1)

    def test_null_table_configuration(self) -> None:
        """Test that a table having null configuration does not break."""
        config = {
            "tables": {
                "model": None,
            }
        }
        with self._get_cmd(config) as gc:
            table = "model"
            gc.do_next(f"{table}.name")
            gc.do_propose("")
            gc.do_set("1")
            gc.do_quit("")
            self.assertEqual(len(gc.config["tables"][table]["row_generators"]), 1)

    def test_prompts(self) -> None:
        """Test that the prompts follow the names of the columns and assigned generators."""
        config: MutableMapping[str, Any] = {}
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
            table = "model"
            column = "name"
            generator = "person.first_name"
            gc.do_next(f"{table}.{column}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[f"generic.{generator}"][0]))
            gc.do_quit("")
            self.assertEqual(len(gc.config["tables"][table]["row_generators"]), 1)
            self.assertDictEqual(
                gc.config["tables"][table]["row_generators"][0],
                {"name": f"generic.{generator}", "columns_assigned": [column]},
            )

    def test_set_generator_distribution(self) -> None:
        """Test that we can set one generator to gaussian."""
        with self._get_cmd({}) as gc:
            table = "string"
            column = "frequency"
            generator = "dist_gen.normal"
            gc.do_next(f"{table}.{column}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[generator][0]))
            gc.do_quit("")
            row_gens = gc.config["tables"][table]["row_generators"]
            self.assertEqual(len(row_gens), 1)
            row_gen = row_gens[0]
            self.assertEqual(row_gen["name"], generator)
            self.assertListEqual(row_gen["columns_assigned"], [column])
            self.assertDictEqual(
                row_gen["kwargs"],
                {
                    "mean": f'SRC_STATS["auto__{table}"]["results"][0]["mean__{column}"]',
                    "sd": f'SRC_STATS["auto__{table}"]["results"][0]["stddev__{column}"]',
                },
            )
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertSetEqual(
                set(gc.config["src-stats"][0].keys()), {"comments", "name", "query"}
            )
            self.assertEqual(gc.config["src-stats"][0]["name"], f"auto__{table}")
            self.assertEqual(
                gc.config["src-stats"][0]["query"],
                (
                    f"SELECT AVG({column}) AS mean__{column}, STDDEV({column})"
                    f" AS stddev__{column} FROM {table}"
                ),
            )

    def test_set_generator_distribution_directly(self) -> None:
        """Test that we can set one generator to gaussian without going through propose."""
        with self._get_cmd({}) as gc:
            table = "string"
            column = "frequency"
            generator = "dist_gen.normal"
            gc.do_next(f"{table}.{column}")
            gc.reset()
            gc.do_set(generator)
            self.assertListEqual(gc.messages, [])
            gc.do_quit("")
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertSetEqual(
                set(gc.config["src-stats"][0].keys()), {"comments", "name", "query"}
            )
            self.assertEqual(gc.config["src-stats"][0]["name"], f"auto__{table}")
            self.assertEqual(
                gc.config["src-stats"][0]["query"],
                (
                    f"SELECT AVG({column}) AS mean__{column}, STDDEV({column})"
                    f" AS stddev__{column} FROM {table}"
                ),
            )

    def test_set_generator_choice(self) -> None:
        """Test that we can set one generator to uniform choice."""
        with self._get_cmd({}) as gc:
            table = "string"
            column = "frequency"
            generator = "dist_gen.choice"
            gc.do_next(f"{table}.{column}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[generator][0]))
            gc.do_quit("")
            row_gens = gc.config["tables"][table]["row_generators"]
            self.assertEqual(len(row_gens), 1)
            row_gen = row_gens[0]
            self.assertEqual(row_gen["name"], generator)
            self.assertListEqual(row_gen["columns_assigned"], [column])
            self.assertDictEqual(
                row_gen["kwargs"],
                {
                    "a": f'SRC_STATS["auto__{table}__{column}"]["results"]',
                },
            )
            self.assertEqual(len(gc.config["src-stats"]), 1)
            self.assertSetEqual(
                set(gc.config["src-stats"][0].keys()), {"comments", "name", "query"}
            )
            self.assertEqual(
                gc.config["src-stats"][0]["name"], f"auto__{table}__{column}"
            )
            self.assertEqual(
                gc.config["src-stats"][0]["query"],
                (
                    f"SELECT {column} AS value FROM {table}"
                    f" WHERE {column} IS NOT NULL"
                    f" GROUP BY value ORDER BY COUNT({column}) DESC"
                ),
            )

    def test_weighted_choice_generator_generates_choices(self) -> None:
        """Test that propose and compare show weighted_choice's values."""
        with self._get_cmd({}) as gc:
            table = "string"
            column = "position"
            generator = "dist_gen.weighted_choice"
            values = {1, 2, 3, 4, 5, 6}
            gc.do_next(f"{table}.{column}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gen_proposal = proposals[generator]
            self.assert_subset(set(gen_proposal[2]), {str(v) for v in values})
            gc.do_compare(str(gen_proposal[0]))
            col_heading = f"{gen_proposal[0]}. {generator}"
            self.assertIn(col_heading, gc.columns)
            self.assert_subset(set(gc.columns[col_heading]), values)

    def test_merge_columns(self) -> None:
        """Test that we can merge columns and set a multivariate generator"""
        table = "string"
        column_1 = "frequency"
        column_2 = "position"
        generator_to_discard = "dist_gen.choice"
        generator = "dist_gen.multivariate_normal"
        with self._get_cmd({}) as gc:
            gc.do_next(f"{table}.{column_2}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            # set a generator, but this should not exist after merging
            gc.do_set(str(proposals[generator_to_discard][0]))
            gc.do_next(f"{table}.{column_1}")
            self.assertIn(table, gc.prompt)
            self.assertIn(column_1, gc.prompt)
            self.assertNotIn(column_2, gc.prompt)
            gc.do_propose("")
            proposals = gc.get_proposals()
            # set a generator, but this should not exist either
            gc.do_set(str(proposals[generator_to_discard][0]))
            gc.do_previous("")
            self.assertIn(table, gc.prompt)
            self.assertIn(column_1, gc.prompt)
            self.assertNotIn(column_2, gc.prompt)
            gc.do_merge(column_2)
            self.assertIn(table, gc.prompt)
            self.assertIn(column_1, gc.prompt)
            self.assertIn(column_2, gc.prompt)
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[generator][0]))
            gc.do_quit("")
            row_gens = gc.config["tables"][table]["row_generators"]
            self.assertEqual(len(row_gens), 1)
            row_gen = row_gens[0]
            self.assertEqual(row_gen["name"], generator)
            self.assertListEqual(row_gen["columns_assigned"], [column_1, column_2])

    def test_unmerge_columns(self) -> None:
        """Test that we can unmerge columns and generators are removed"""
        table = "string"
        column_1 = "frequency"
        column_2 = "position"
        column_3 = "model_id"
        remaining_gen = "gen3"
        config = {
            "tables": {
                table: {
                    "row_generators": [
                        {"name": "gen1", "columns_assigned": [column_1, column_2]},
                        {"name": remaining_gen, "columns_assigned": [column_3]},
                    ]
                }
            }
        }
        with self._get_cmd(config) as gc:
            gc.do_next(f"{table}.{column_2}")
            self.assertIn(table, gc.prompt)
            self.assertIn(column_1, gc.prompt)
            self.assertIn(column_2, gc.prompt)
            gc.do_unmerge(column_1)
            self.assertIn(table, gc.prompt)
            self.assertNotIn(column_1, gc.prompt)
            self.assertIn(column_2, gc.prompt)
            # Next generator should be the unmerged one
            gc.do_next("")
            self.assertIn(table, gc.prompt)
            self.assertIn(column_1, gc.prompt)
            self.assertNotIn(column_2, gc.prompt)
            gc.do_quit("")
            # Both generators should have disappeared
            row_gens = gc.config["tables"][table]["row_generators"]
            self.assertEqual(len(row_gens), 1)
            row_gen = row_gens[0]
            self.assertEqual(row_gen["name"], remaining_gen)
            self.assertListEqual(row_gen["columns_assigned"], [column_3])

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
                    "query": (
                        "SELECT AVG(frequency) AS mean__frequency,"
                        " STDDEV(frequency) AS stddev__frequency FROM string"
                    ),
                }
            ],
        }
        with self._get_cmd(config) as gc:
            table = "model"
            column = "name"
            generator = "person.first_name"
            gc.do_next(f"{table}.{column}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[f"generic.{generator}"][0]))
            gc.do_quit("")
            self.assertEqual(len(gc.config["tables"][table]["row_generators"]), 1)
            self.assertDictEqual(
                gc.config["tables"][table]["row_generators"][0],
                {"name": f"generic.{generator}", "columns_assigned": [column]},
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
            self.assertEqual(gc.config["src-stats"][0]["name"], "auto__string")
            self.assertEqual(
                gc.config["src-stats"][0]["query"],
                (
                    "SELECT AVG(frequency) AS mean__frequency,"
                    " STDDEV(frequency) AS stddev__frequency FROM string"
                ),
            )

    def test_aggregate_queries_merge(self) -> None:
        """
        Test that we can set a generator that requires select aggregate clauses
        and keep an old one, resulting in a merged query.
        """
        rg = {
            "name": "dist_gen.normal",
            "columns_assigned": ["frequency"],
            "kwargs": {
                "mean": 'SRC_STATS["auto__string"]["results"][0]["mean__frequency"]',
                "sd": 'SRC_STATS["auto__string"]["results"][0]["stddev__frequency"]',
            },
        }
        config = {
            "tables": {"string": {"row_generators": [rg]}},
            "src-stats": [
                {
                    "name": "auto__string",
                    "query": (
                        "SELECT AVG(frequency) AS mean__frequency,"
                        " STDDEV(frequency) AS stddev__frequency FROM string"
                    ),
                }
            ],
        }
        with self._get_cmd(copy.deepcopy(config)) as gc:
            column = "position"
            generator = "dist_gen.uniform_ms"
            gc.do_next(f"string.{column}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[f"{generator}"][0]))
            gc.do_quit("")
            row_gens: list[dict[str, Any]] = gc.config["tables"]["string"][
                "row_generators"
            ]
            self.assertEqual(len(row_gens), 2)
            if row_gens[0]["name"] == generator:
                row_gen0 = row_gens[0]
                row_gen1 = row_gens[1]
            else:
                row_gen0 = row_gens[1]
                row_gen1 = row_gens[0]
            self.assertEqual(row_gen0["name"], generator)
            self.assertEqual(row_gen1["name"], "dist_gen.normal")
            self.assertListEqual(row_gen0["columns_assigned"], [column])
            self.assertDictEqual(
                row_gen0["kwargs"],
                {
                    "mean": f'SRC_STATS["auto__string"]["results"][0]["mean__{column}"]',
                    "sd": f'SRC_STATS["auto__string"]["results"][0]["stddev__{column}"]',
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
            assert (
                select_match is not None
            ), "src_stats[0].query is not an aggregate select"
            self.assertSetEqual(
                set(select_match.group(1).split(", ")),
                {
                    "AVG(frequency) AS mean__frequency",
                    "STDDEV(frequency) AS stddev__frequency",
                    f"AVG({column}) AS mean__{column}",
                    f"STDDEV({column}) AS stddev__{column}",
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
        config: MutableMapping[str, Any] = {
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
            column = "position"
            generator = "dist_gen.uniform_ms"
            gc.do_next(f"string.{column}")
            gc.do_propose("")
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[f"{generator}"][0]))
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

    def setUp(self) -> None:
        super().setUp()
        ChoiceGeneratorFactory.SAMPLE_COUNT = 500
        ChoiceGeneratorFactory.SUPPRESS_COUNT = 5

    def _get_cmd(self, config: MutableMapping[str, Any]) -> TestGeneratorCmd:
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
        with self.sync_engine.connect() as conn:
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
        with self.sync_engine.connect() as conn:
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
            self.assert_subset(set(prop[2]), {"1", "4"})
            gc.reset()
            gc.do_compare(str(prop[0]))
            col_heading = (
                f"{prop[0]}. dist_gen.weighted_choice [sampled and suppressed]"
            )
            self.assertIn(col_heading, set(gc.columns.keys()))
            col_set: set[int] = set(gc.columns[col_heading])
            self.assert_subset(col_set, {1, 4})
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
            self.assert_subset(set(prop[2]), {"1", "2", "3", "4", "5"})
            gc.reset()
            gc.do_compare(str(prop[0]))
            col_heading = f"{prop[0]}. dist_gen.weighted_choice"
            self.assertIn(col_heading, set(gc.columns.keys()))
            col_set2: set[int] = set(gc.columns[col_heading])
            self.assert_subset(col_set2, {1, 2, 3, 4, 5})
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
            self.assert_subset(set(prop[2]), {"1", "2", "3", "4", "5"})
            gc.do_compare(str(prop[0]))
            col_heading = f"{prop[0]}. dist_gen.weighted_choice [sampled]"
            self.assertIn(col_heading, set(gc.columns.keys()))
            col_set3: set[int] = set(gc.columns[col_heading])
            self.assert_subset(col_set3, {1, 2, 3, 4, 5})
            gc.do_set(str(prop[0]))
            gc.do_quit("")
            self.generate_data(gc.config, num_passes=200)
        with self.sync_engine.connect() as conn:
            ones = set()
            twos = set()
            threes = set()
            for row in conn.execute(
                select(self.metadata.tables["number_table"])
            ).fetchall():
                ones.add(row.one)
                twos.add(row.two)
                threes.add(row.three)
            # all generation possibilities should be present
            self.assertSetEqual(ones, {1, 4})
            self.assertSetEqual(twos, {1, 2, 3, 4, 5})
            self.assertSetEqual(threes, {1, 2, 3, 4, 5})


class GeneratorTests(GeneratesDBTestCase):
    """Testing configure-generators with generation."""

    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    def _get_cmd(self, config: MutableMapping[str, Any]) -> TestGeneratorCmd:
        """We are using configure-generators."""
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
        with self.sync_engine.connect() as conn:
            # select(self.metadata.tables["string"].c["position", "frequency"]) would be nicer
            # but mypy doesn't like it
            stmt = select(
                self.metadata.tables["string"].c["position"],
                self.metadata.tables["string"].c["frequency"],
            )
            rows = conn.execute(stmt).fetchall()
            count = 0
            for row in rows:
                count += 1
                self.assertEqual(row.position, 0)
                self.assertEqual(row.frequency, 0.0)
            self.assertEqual(count, 3)
            # select(self.metadata.tables["signature_model"].c["name", "based_on"]) would be nicer
            # but mypy doesn't like it
            stmt = select(
                self.metadata.tables["signature_model"].c["name"],
                self.metadata.tables["signature_model"].c["based_on"],
            )
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

    def assert_are_truncated_to(self, xs: Iterable[str], length: int) -> None:
        """
        Check that none of the strings are longer than ``length`` (after
        removing surrounding quotes).
        """
        maxlen = 0
        for x in xs:
            newlen = len(x.strip("'\""))
            self.assertLessEqual(newlen, length)
            maxlen = max(maxlen, newlen)
        self.assertEqual(maxlen, length)

    def test_varchar_ns_are_truncated(self) -> None:
        """Tests that mimesis generators for VARCHAR(N) truncate to N characters"""
        generator = "generic.text.quote"
        table = "signature_model"
        column = "name"
        with self._get_cmd({}) as gc:
            gc.do_next(f"{table}.{column}")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            quotes = [k for k in proposals.keys() if k.startswith(generator)]
            self.assertEqual(len(quotes), 1)
            prop = proposals[quotes[0]]
            self.assert_are_truncated_to(prop[2], 20)
            gc.reset()
            gc.do_compare(str(prop[0]))
            col_heading = f"{prop[0]}. {quotes[0]}"
            gc.do_set(str(prop[0]))
            self.assertIn(col_heading, gc.columns)
            self.assert_are_truncated_to(gc.columns[col_heading], 20)
            gc.do_quit("")
            config = gc.config
            self.generate_data(config, num_passes=15)
        with self.sync_engine.connect() as conn:
            stmt = select(self.metadata.tables[table].c[column])
            rows = conn.execute(stmt).scalars().fetchall()
            self.assert_are_truncated_to(rows, 20)


@dataclass
class Stat:
    """Mean and variance calculator."""

    n: int = 0
    x: float = 0
    x2: float = 0

    def add(self, x: float) -> None:
        """Add one datum."""
        self.n += 1
        self.x += x
        self.x2 += x * x

    def count(self) -> int:
        """Get the number of data added."""
        return self.n

    def x_mean(self) -> float:
        """Get the mean of the added data."""
        return self.x / self.n

    def x_var(self) -> float:
        """Get the variance of the added data."""
        x = self.x
        return (self.x2 - x * x / self.n) / (self.n - 1)


@dataclass
class Correlation(Stat):
    """Mean, variance and covariance."""

    y: float = 0
    y2: float = 0
    xy: float = 0

    def add2(self, x: float, y: float) -> None:
        """Add a 2D data point."""
        self.n += 1
        self.x += x
        self.x2 += x * x
        self.y += y
        self.y2 += y * y
        self.xy += x * y

    def y_mean(self) -> float:
        """Get the mean of the second parts of the added points."""
        return self.y / self.n

    def y_var(self) -> float:
        """Get the variance of the second parts of the added points."""
        y = self.y
        return (self.y2 - y * y / self.n) / (self.n - 1)

    def covar(self) -> float:
        """Get the covariance of the two parts of the added points."""
        return (self.xy - self.x * self.y / self.n) / (self.n - 1)


# pylint disable: too-many-instance-attributes
class EavMeasurementTableStats:
    """The statistics for the Measurement table of eav.sql."""

    def __init__(self, conn: Connection, metadata: MetaData, test: TestCase) -> None:
        stmt = select(metadata.tables["measurement"])
        rows = conn.execute(stmt).fetchall()
        self.types: set[int] = set()
        self.one_count = 0
        self.one_yes_count = 0
        self.two = Correlation()
        self.three = Correlation()
        self.four = Correlation()
        self.fish = Stat()
        self.fowl = Stat()
        for row in rows:
            self.types.add(row.type)
            if row.type == 1:
                # yes or no
                test.assertIsNone(row.first_value)
                test.assertIsNone(row.second_value)
                test.assertIn(row.third_value, {"yes", "no"})
                self.one_count += 1
                if row.third_value == "yes":
                    self.one_yes_count += 1
            elif row.type == 2:
                # positive correlation around 1.4, 1.8
                test.assertIsNotNone(row.first_value)
                test.assertIsNotNone(row.second_value)
                test.assertIsNone(row.third_value)
                self.two.add2(row.first_value, row.second_value)
            elif row.type == 3:
                # negative correlation around 11.8, 12.1
                test.assertIsNotNone(row.first_value)
                test.assertIsNotNone(row.second_value)
                test.assertIsNone(row.third_value)
                self.three.add2(row.first_value, row.second_value)
            elif row.type == 4:
                # positive correlation around 21.4, 23.4
                test.assertIsNotNone(row.first_value)
                test.assertIsNotNone(row.second_value)
                test.assertIsNone(row.third_value)
                self.four.add2(row.first_value, row.second_value)
            elif row.type == 5:
                test.assertIn(row.third_value, {"fish", "fowl"})
                test.assertIsNotNone(row.first_value)
                test.assertIsNone(row.second_value)
                if row.third_value == "fish":
                    # mean 8.1 and sd 0.755
                    self.fish.add(row.first_value)
                else:
                    # mean 11.2 and sd 1.114
                    self.fowl.add(row.first_value)


class NullPartitionedTests(GeneratesDBTestCase):
    """Testing null-partitioned grouped multivariate generation."""

    dump_file_path = "eav.sql"
    database_name = "eav"
    schema_name = "public"

    def setUp(self) -> None:
        """Set up the test with specific sample and suppress counts."""
        super().setUp()
        NullPartitionedNormalGeneratorFactory.SAMPLE_COUNT = 8
        NullPartitionedNormalGeneratorFactory.SUPPRESS_COUNT = 2

    def _get_cmd(self, config: MutableMapping[str, Any]) -> TestGeneratorCmd:
        """Get the configure-generators object as our command."""
        return TestGeneratorCmd(self.dsn, self.schema_name, self.metadata, config)

    def test_create_with_null_partitioned_grouped_multivariate(self) -> None:
        """Test EAV for all columns."""
        generate_count = 800
        with self._get_cmd({}) as gc:
            self.merge_columns(
                gc,
                "measurement",
                [
                    "type",
                    "first_value",
                    "second_value",
                    "third_value",
                ],
            )
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
            self.populate_measurement_type_vocab()
            self.create_data(gc.config, num_passes=generate_count)
        with self.sync_engine.connect() as conn:
            stats = EavMeasurementTableStats(conn, self.metadata, self)
        # type 1
        self.assertAlmostEqual(
            stats.one_count, generate_count * 5 / 20, delta=generate_count * 0.4
        )
        # about 40% are yes
        self.assertAlmostEqual(
            stats.one_yes_count / stats.one_count, 0.4, delta=generate_count * 0.4
        )
        # type 2
        self.assertAlmostEqual(
            stats.two.count(), generate_count * 3 / 20, delta=generate_count * 0.5
        )
        self.assertAlmostEqual(stats.two.x_mean(), 1.4, delta=0.4)
        self.assertAlmostEqual(stats.two.x_var(), 0.315, delta=0.18)
        self.assertAlmostEqual(stats.two.y_mean(), 1.8, delta=0.8)
        self.assertAlmostEqual(stats.two.y_var(), 0.105, delta=0.06)
        self.assertAlmostEqual(stats.two.covar(), 0.105, delta=0.07)
        # type 3
        self.assertAlmostEqual(
            stats.three.count(), generate_count * 3 / 20, delta=generate_count * 0.2
        )
        self.assertAlmostEqual(stats.three.covar(), -2.085, delta=1.1)
        # type 4
        self.assertAlmostEqual(
            stats.four.count(), generate_count * 3 / 20, delta=generate_count * 0.2
        )
        self.assertAlmostEqual(stats.four.covar(), 3.33, delta=1)
        # type 5/fish
        self.assertAlmostEqual(
            stats.fish.count(), generate_count * 3 / 20, delta=generate_count * 0.2
        )
        self.assertAlmostEqual(stats.fish.x_mean(), 8.1, delta=3.0)
        self.assertAlmostEqual(stats.fish.x_var(), 0.855, delta=0.6)
        # type 5/fowl
        self.assertAlmostEqual(
            stats.fowl.count(), generate_count * 3 / 20, delta=generate_count * 0.2
        )
        self.assertAlmostEqual(stats.fowl.x_mean(), 11.2, delta=8.0)
        self.assertAlmostEqual(stats.fowl.x_var(), 1.86, delta=1)

    def populate_measurement_type_vocab(self):
        """Add a vocab table without messing around with files"""
        table = self.metadata.tables["measurement_type"]
        with self.engine.connect() as conn:
            conn.execute(insert(table).values({"id": 1, "name": "agreement"}))
            conn.execute(insert(table).values({"id": 2, "name": "acceleration"}))
            conn.execute(insert(table).values({"id": 3, "name": "velocity"}))
            conn.execute(insert(table).values({"id": 4, "name": "position"}))
            conn.execute(insert(table).values({"id": 5, "name": "matter"}))
            conn.commit()

    def merge_columns(
        self, gc: TestGeneratorCmd, table: str, columns: list[str]
    ) -> None:
        """Merge columns in a table"""
        gc.do_next(f"{table}.{columns[0]}")
        for col in columns[1:]:
            gc.do_merge(col)
        gc.reset()

    def test_create_with_null_partitioned_grouped_sampled_and_suppressed(self) -> None:
        """Test EAV for all columns with sampled and suppressed generation."""
        generate_count = 800
        with self._get_cmd({}) as gc:
            self.merge_columns(
                gc,
                "measurement",
                [
                    "type",
                    "first_value",
                    "second_value",
                    "third_value",
                ],
            )
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
            self.merge_columns(
                gc,
                "observation",
                [
                    "type",
                    "first_value",
                    "second_value",
                    "third_value",
                ],
            )
            gc.do_propose("")
            proposals = gc.get_proposals()
            prop = proposals[dist_to_choose]
            gc.do_set(str(prop[0]))
            gc.do_quit("")
            self.set_configuration(gc.config)
            self.get_src_stats(gc.config)
            self.create_generators(gc.config)
            self.remove_data(gc.config)
            self.populate_measurement_type_vocab()
            self.create_data(gc.config, num_passes=generate_count)
        with self.sync_engine.connect() as conn:
            stats = EavMeasurementTableStats(conn, self.metadata, self)
            stmt = select(self.metadata.tables["observation"])
            rows = conn.execute(stmt).fetchall()
            firsts = Stat()
            for row in rows:
                stats.types.add(row.type)
                self.assertEqual(row.type, 1)
                self.assertIsNotNone(row.first_value)
                self.assertIsNone(row.second_value)
                self.assertIn(row.third_value, {"ham", "eggs"})
                firsts.add(row.first_value)
            self.assertEqual(firsts.count(), 800)
            self.assertAlmostEqual(firsts.x_mean(), 1.3, delta=generate_count * 0.3)
        self.assert_subset(stats.types, {1, 2, 3, 4, 5})
        self.assertEqual(len(stats.types), 4)
        self.assert_subset({1, 5}, stats.types)
        # type 1
        self.assertAlmostEqual(
            stats.one_count, generate_count * 5 / 11, delta=generate_count * 0.4
        )
        # about 40% are yes
        self.assertAlmostEqual(
            stats.one_yes_count / stats.one_count, 0.4, delta=generate_count * 0.4
        )
        # type 5/fish
        self.assertAlmostEqual(
            stats.fish.count(), generate_count * 3 / 11, delta=generate_count * 0.2
        )
        self.assertAlmostEqual(stats.fish.x_mean(), 8.1, delta=3.0)
        self.assertAlmostEqual(stats.fish.x_var(), 0.855, delta=0.5)
        # type 5/fowl
        self.assertAlmostEqual(
            stats.fowl.count(), generate_count * 3 / 11, delta=generate_count * 0.2
        )
        self.assertAlmostEqual(stats.fowl.x_mean(), 11.2, delta=8.0)
        self.assertAlmostEqual(stats.fowl.x_var(), 1.86, delta=1)

    def test_create_with_null_partitioned_grouped_sampled_only(self):
        """Test EAV for all columns with sampled generation but no suppression."""
        table_name = "measurement"
        table2_name = "observation"
        generate_count = 800
        with self._get_cmd({}) as gc:
            self.merge_columns(
                gc, table_name, ["type", "first_value", "second_value", "third_value"]
            )
            gc.do_propose("")
            proposals = gc.get_proposals()
            self.assertIn("null-partitioned grouped_multivariate_lognormal", proposals)
            self.assertIn("null-partitioned grouped_multivariate_normal", proposals)
            self.assertIn(
                "null-partitioned grouped_multivariate_lognormal [sampled and suppressed]",
                proposals,
            )
            self.assertIn(
                "null-partitioned grouped_multivariate_normal [sampled and suppressed]",
                proposals,
            )
            self.assertIn(
                "null-partitioned grouped_multivariate_lognormal [sampled]", proposals
            )
            dist_to_choose = "null-partitioned grouped_multivariate_normal [sampled]"
            self.assertIn(dist_to_choose, proposals)
            prop = proposals[dist_to_choose]
            gc.reset()
            gc.do_compare(str(prop[0]))
            col_heading = f"{prop[0]}. {dist_to_choose}"
            self.assertIn(col_heading, set(gc.columns.keys()))
            gc.do_set(str(prop[0]))
            self.merge_columns(
                gc, table2_name, ["type", "first_value", "second_value", "third_value"]
            )
            gc.do_propose("")
            proposals = gc.get_proposals()
            prop = proposals[dist_to_choose]
            gc.do_set(str(prop[0]))
            gc.do_quit("")
            self.set_configuration(gc.config)
            self.get_src_stats(gc.config)
            self.create_generators(gc.config)
            self.remove_data(gc.config)
            self.populate_measurement_type_vocab()
            self.create_data(gc.config, num_passes=generate_count)
        with self.engine.connect() as conn:
            stmt = select(self.metadata.tables[table_name])
            rows = conn.execute(stmt).fetchall()
            self.assert_subset({row.type for row in rows}, {1, 2, 3, 4, 5})
            stmt = select(self.metadata.tables[table2_name])
            rows = conn.execute(stmt).fetchall()
            self.assertEqual(
                {row.third_value for row in rows}, {"ham", "eggs", "cheese"}
            )

    def test_create_with_null_partitioned_grouped_sampled_tiny(self):
        """
        Test EAV for all columns with sampled generation that only gets a tiny sample.
        """
        # five will ensure that at least one group will have two elements in it,
        # but all three cannot.
        NullPartitionedNormalGeneratorFactory.SAMPLE_COUNT = 5
        table_name = "observation"
        generate_count = 100
        with self._get_cmd({}) as gc:
            dist_to_choose = "null-partitioned grouped_multivariate_normal [sampled]"
            self.merge_columns(
                gc, table_name, ["type", "first_value", "second_value", "third_value"]
            )
            gc.do_propose("")
            proposals = gc.get_proposals()
            # breakpoint()
            prop = proposals[dist_to_choose]
            gc.do_set(str(prop[0]))
            gc.do_quit("")
            self.set_configuration(gc.config)
            self.get_src_stats(gc.config)
            self.create_generators(gc.config)
            self.remove_data(gc.config)
            self.populate_measurement_type_vocab()
            self.create_data(gc.config, num_passes=generate_count)
        with self.engine.connect() as conn:
            stmt = select(self.metadata.tables[table_name])
            rows = conn.execute(stmt).fetchall()
            # we should only have one or two of "ham", "eggs" and "cheese" represented
            foods = {row.third_value for row in rows}
            self.assert_subset(foods, {"ham", "eggs", "cheese"})
            self.assertLess(len(foods), 3)


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
        config: MutableMapping[str, Any] = {}
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
        self, mock_csv_reader: MagicMock, mock_path: MagicMock
    ):
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
    def test_non_interactive_configure_null_partitioned_where_existing_merges(
        self, _mock_csv_reader: MagicMock, _mock_path: MagicMock
    ) -> None:
        """
        test that we can set multi-column generators from a CSV file,
        but where there are already multi-column generators configured
        that will have to be unmerged.
        """
        config = {
            "tables": {
                "observation": {
                    "row_generators": [
                        {
                            "name": "arbitrary_gen",
                            "columns_assigned": [
                                "type",
                                "second_value",
                                "first_value",
                            ],
                        }
                    ],
                },
            },
        }
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
