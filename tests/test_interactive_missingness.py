""" Tests for the configure-missingness command. """
import random
from collections.abc import MutableMapping
from typing import Any

from sqlalchemy import select

from datafaker.interactive import MissingnessCmd
from tests.utils import GeneratesDBTestCase, RequiresDBTestCase, TestDbCmdMixin


class TestMissingnessCmd(MissingnessCmd, TestDbCmdMixin):
    """MissingnessCmd but mocked"""


class ConfigureMissingnessTests(RequiresDBTestCase):
    """Testing configure-missing."""

    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    def _get_cmd(self, config: MutableMapping[str, Any]) -> TestMissingnessCmd:
        """We are using configure-missingness."""
        return TestMissingnessCmd(self.dsn, self.schema_name, self.metadata, config)

    def test_set_missingness_to_sampled(self) -> None:
        """Test that we can set one table to sampled missingness."""
        with self._get_cmd({}) as mc:
            table = "signature_model"
            mc.do_next(table)
            mc.do_counts("")
            self.assertSequenceEqual(
                mc.messages, [(MissingnessCmd.ROW_COUNT_MSG, (10,), {})]
            )
            # Check the counts of NULLs in each column
            self.assertSequenceEqual(mc.rows, [["player_id", 4], ["based_on", 3]])
            mc.do_sampled("")
            mc.do_quit("")
            self.assertListEqual(
                mc.config["tables"][table]["missingness_generators"],
                [
                    {
                        "columns": ["player_id", "based_on"],
                        "kwargs": {
                            "patterns": 'SRC_STATS["missing_auto__signature_model__0"]["results"]'
                        },
                        "name": "column_presence.sampled",
                    }
                ],
            )
            self.assertEqual(
                mc.config["src-stats"][0]["name"],
                "missing_auto__signature_model__0",
            )
            self.assertEqual(
                mc.config["src-stats"][0]["query"],
                (
                    "SELECT COUNT(*) AS row_count,"
                    " player_id__is_null, based_on__is_null FROM"
                    " (SELECT player_id IS NULL AS player_id__is_null,"
                    " based_on IS NULL AS based_on__is_null FROM"
                    " signature_model ORDER BY RANDOM() LIMIT 1000)"
                    " AS __t GROUP BY player_id__is_null, based_on__is_null"
                ),
            )


class ConfigureMissingnessTestsWithGeneration(GeneratesDBTestCase):
    """Testing configure-missing with generation."""

    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    def _get_cmd(self, config: MutableMapping[str, Any]) -> TestMissingnessCmd:
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
        with self.sync_engine.connect() as conn:
            stmt = select(self.metadata.tables[table_name])
            rows = conn.execute(stmt).mappings().fetchall()
            patterns: set[int] = set()
            for row in rows:
                p = 0 if row["player_id"] is None else 1
                b = 0 if row["based_on"] is None else 2
                patterns.add(p + b)
            # all pattern possibilities should be present
            self.assertSetEqual(patterns, {0, 1, 2, 3})
