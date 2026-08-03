""" Tests for the configure-generators command. """
import copy
import re
import sys
from collections.abc import MutableMapping
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

import yaml
from sqlalchemy import func, select

from datafaker.dialects import SecondsDifference, StdDev
from datafaker.interactive.base import DbCmd
from datafaker.interactive.generators import GeneratorCmd
from datafaker.theme import ThemeEntry, set_active_theme
from tests.utils import (
    DuckTestDb,
    GeneratesDBTestCase,
    MsSqlTestDb,
    RequiresDBTestCase,
    TestDbCmdMixin,
)


class MockGeneratorCmd(GeneratorCmd, TestDbCmdMixin):
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

    def setUp(self) -> None:
        super().setUp()
        set_active_theme(ThemeEntry.NONE)

    def _get_cmd(self, config: MutableMapping[str, Any]) -> MockGeneratorCmd:
        """Get the command we are using for this test case."""
        return MockGeneratorCmd(
            DbCmd.Settings(self.dsn, self.schema_name, config, self.metadata, None)
        )

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
                    f"SELECT avg({table}.{column}) AS mean__{column}, STDDEV({table}.{column})"
                    f' AS stddev__{column} FROM "{table}"'
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
                    f"SELECT avg({table}.{column}) AS mean__{column}, STDDEV({table}.{column})"
                    f' AS stddev__{column} FROM "{table}"'
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
                    "SELECT _counted.value \n"
                    f'FROM (SELECT "{column}" AS value, count("{column}") AS count \n'
                    f"FROM {table} \n"
                    f'WHERE "{column}" IS NOT NULL GROUP BY "{column}") '
                    "AS _counted ORDER BY _counted.count DESC"
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
                    ' STDDEV(frequency) AS stddev__frequency FROM "string"'
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
                r'SELECT (.*) FROM "string"', gc.config["src-stats"][0]["query"]
            )
            assert (
                select_match is not None
            ), "src_stats[0].query is not an aggregate select"
            self.assertSetEqual(
                set(select_match.group(1).split(", ")),
                {
                    "AVG(frequency) AS mean__frequency",
                    "STDDEV(frequency) AS stddev__frequency",
                    f"avg(string.{column}) AS mean__{column}",
                    f"STDDEV(string.{column}) AS stddev__{column}",
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

    def test_select_completion(self) -> None:
        """Test tab completion for the select command."""
        with self._get_cmd({}) as gc:
            # Select clause completes tables and columns
            self.assertSetEqual(
                set(gc.complete_select("m", "select m", 7, 8)),
                {"manufacturer", "model"},
            )
            self.assertSetEqual(
                set(gc.complete_select("model", "select model", 7, 12)),
                {"model", "model."},
            )
            self.assertSetEqual(
                set(gc.complete_select("string.", "select string.", 7, 13)),
                {"string.id", "string.model_id", "string.position", "string.frequency"},
            )
            self.assertSetEqual(
                set(gc.complete_select("string.p", "select string.p", 7, 14)),
                {"string.position"},
            )
            self.assertListEqual(
                gc.complete_select("string.q", "select string.q", 7, 14), []
            )
            self.assertListEqual(gc.complete_next("ww", "select ww", 7, 9), [])
            # From clause completes tables only
            self.assertSetEqual(
                set(gc.complete_select("m", "select id from m", 15, 16)),
                {"manufacturer", "model"},
            )
            self.assertListEqual(
                gc.complete_select("string.p", "select id from string.p", 15, 22),
                [],
            )
            # SQL completions
            self.assertListEqual(
                gc.complete_select("fr", "select id fr", 10, 12), ["from"]
            )
            self.assertListEqual(
                gc.complete_select("FR", "select id FR", 10, 12), ["FROM"]
            )
            self.assertListEqual(
                gc.complete_select("fu", "select id from string fu", 22, 24), ["full"]
            )
            self.assertListEqual(
                gc.complete_select("ou", "select id from string full ou", 25, 27), ["outer"]
            )
            self.assertListEqual(
                gc.complete_select("jo", "select id from string full outer jo", 29, 31), ["join"]
            )

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
            gc.do_set(str(proposals[generator][0]))
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


class ConfigureGeneratorsWithSrc2Tests(GeneratesDBTestCase):
    """Test `configure-generators` with the `src2.dump` database."""

    dump_file_path = "src2.dump"
    database_name = "src"
    schema_name = "public"
    use_temporary_cwd = True
    copy_files = ["row_generators.py", "story_generators.py"]
    copy_from_directory = Path("examples")

    def setUp(self) -> None:
        super().setUp()
        set_active_theme(ThemeEntry.NONE)

    def _get_cmd(self, config: MutableMapping[str, Any]) -> MockGeneratorCmd:
        """Get the command we are using for this test case."""
        return MockGeneratorCmd(
            DbCmd.Settings(self.dsn, self.schema_name, config, self.metadata, None)
        )

    def _get_config(self) -> dict[Any, Any]:
        test_module = resources.files(sys.modules["tests"])
        with test_module.joinpath("examples/example_config2.yaml").open(
            encoding="utf-8",
        ) as config_fh:
            cy = yaml.load(config_fh, yaml.SafeLoader)
            assert isinstance(cy, dict)
            return cy

    def test_intervals_end_to_end(self) -> None:
        """Test that if an interval end is applicable it gets proposed and works."""
        table = "hospital_visit"
        column = "visit_end"
        config = self._get_config()
        # let's not test the uniqueness failures!
        config["tables"]["unique_constraint_test"]["num_rows_per_pass"] = 0
        config["tables"]["unique_constraint_test2"]["num_rows_per_pass"] = 0
        with self._get_cmd(config) as gc:
            # set up our interval proposer
            gc.do_next(f"{table}.{column}")
            gc.do_unmerge("visit_start")
            gc.reset()
            gc.do_propose("")
            proposals = gc.get_proposals()
            provider_name = (
                "generic.anchored_provider.normal_date [anchored to visit_start]"
            )
            self.assertIn(provider_name, proposals)
            proposals = gc.get_proposals()
            gc.do_set(str(proposals[provider_name][0]))
            gc.do_quit("")
            self.generate_data(config, num_passes=15)
        with self.sync_engine.connect() as conn:
            src_diff = SecondsDifference(
                self.metadata.tables[table].c[column],
                self.metadata.tables[table].c["visit_start"],
            )
            src_result = conn.execute(
                select(
                    func.avg(src_diff).label("mean"), StdDev(src_diff).label("sd")
                ).select_from(self.metadata.tables[table])
            ).one()
        assert self.dst_engine is not None
        with self.dst_engine.connect() as conn:
            dst_diff = SecondsDifference(
                self.dst_metadata.tables[table].c[column],
                self.dst_metadata.tables[table].c["visit_start"],
            )
            dst_result = conn.execute(
                select(
                    func.avg(dst_diff).label("mean"), StdDev(dst_diff).label("sd")
                ).select_from(self.dst_metadata.tables[table])
            ).one()
        self.assertAlmostEqual(
            src_result.mean, dst_result.mean, delta=src_result.mean * 0.3
        )
        self.assertAlmostEqual(src_result.sd, dst_result.sd, delta=src_result.sd * 0.5)


class ConfigureGeneratorsWithSrc2DuckDbTests(ConfigureGeneratorsWithSrc2Tests):
    """Test `configure-generators` with `src2.dump` with DuckDB."""

    database_type = DuckTestDb


class ConfigureGeneratorsWithSrc2MsSqlTests(ConfigureGeneratorsWithSrc2Tests):
    """Test `configure-generators` with `src2.dump` with DuckDB."""

    database_type = MsSqlTestDb
    schema_name = None


class GeneratorTests(GeneratesDBTestCase):
    """Testing configure-generators with generation."""

    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    def setUp(self) -> None:
        super().setUp()
        set_active_theme(ThemeEntry.NONE)

    def _get_cmd(self, config: MutableMapping[str, Any]) -> MockGeneratorCmd:
        """We are using configure-generators."""
        return MockGeneratorCmd(
            DbCmd.Settings(self.dsn, self.schema_name, config, self.metadata, None)
        )

    def test_set_null(self) -> None:
        """Test that we can sample real missingness and reproduce it."""
        with self._get_cmd({}) as gc:
            gc.do_next("string.position")
            gc.do_set("dist_gen.constant")
            gc.do_next("string.frequency")
            gc.do_set("dist_gen.constant")
            gc.do_next("signature_model.name")
            gc.do_set("dist_gen.constant")
            gc.do_next("signature_model.based_on")
            gc.do_set("dist_gen.constant")
            # We should have got no errors, but one of these will have been
            # the last table and so would have produced the "no more tables" error
            self.assertListEqual(
                gc.messages, [(GeneratorCmd.INFO_NO_MORE_TABLES, (), {})]
            )
            gc.reset()
            gc.do_quit("")
            config = gc.config
            self.generate_data(config, num_passes=3)
        # Test that each missingness pattern is present in the database
        assert self.dst_engine is not None
        with self.dst_engine.connect() as conn:
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
        if self.database_type is DuckTestDb:
            # DuckDB does not support limited width VARCHARs
            return
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
        assert self.dst_engine is not None
        with self.dst_engine.connect() as conn:
            stmt = select(self.metadata.tables[table].c[column])
            rows = conn.execute(stmt).scalars().fetchall()
            self.assert_are_truncated_to(rows, 20)


class GeneratorTestsDuckDb(GeneratorTests):
    """As ``GeneratorTests`` but with DuckDB."""

    database_type = DuckTestDb


class GeneratorTestsMsSql(GeneratorTests):
    """As ``GeneratorTests`` but with M SSql."""

    database_type = MsSqlTestDb
    schema_name = None
