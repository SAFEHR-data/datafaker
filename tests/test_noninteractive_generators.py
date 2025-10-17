""" Tests for the configure-generators command with the --spec option. """

from collections.abc import Mapping, MutableMapping
from typing import Any
from unittest.mock import MagicMock, Mock, patch

from datafaker.interactive import update_config_generators
from tests.utils import RequiresDBTestCase


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
        self, _mock_csv_reader: MagicMock, _mock_path: MagicMock
    ) -> None:
        """
        test that we can set multi-column generators from a CSV file
        """
        config: MutableMapping[str, Any] = {}
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
        row_gens: Mapping[str, Any] = {
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
