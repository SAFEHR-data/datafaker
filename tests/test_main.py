"""Tests for the main module."""
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import yaml
from click.testing import Result
from typer.testing import CliRunner

from datafaker.main import app
from datafaker.settings import Settings
from tests.utils import DatafakerTestCase, get_test_settings

runner = CliRunner(mix_stderr=False)


class TestCLI(DatafakerTestCase):
    """Tests for the command-line interface."""

    @patch("datafaker.main.read_config_file")
    @patch("datafaker.main.dict_to_metadata")
    @patch("datafaker.main.load_metadata_config")
    @patch("datafaker.main.create_db_vocab")
    def test_create_vocab(self, mock_create: MagicMock, mock_mdict: MagicMock, mock_meta: MagicMock, mock_config: MagicMock) -> None:
        """Test the create-vocab sub-command."""
        result = runner.invoke(
            app,
            [
                "create-vocab",
            ],
            catch_exceptions=False,
        )

        mock_create.assert_called_once_with(mock_meta.return_value, mock_mdict.return_value, mock_config.return_value)
        self.assertSuccess(result)

    @patch("datafaker.main.read_config_file")
    @patch("datafaker.main.load_metadata")
    @patch("datafaker.main.get_settings")
    @patch("datafaker.main.Path")
    @patch("datafaker.main.make_table_generators")
    @patch("datafaker.main.generators_require_stats")
    def test_create_generators(
        self,
        mock_require_stats: MagicMock,
        mock_make: MagicMock,
        mock_path: MagicMock,
        mock_settings: MagicMock,
        mock_load_meta: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test the create-generators sub-command."""
        mock_require_stats.return_value = False
        mock_path.return_value.exists.return_value = False
        mock_make.return_value = "some text"
        mock_settings.return_value.src_postges_dsn = ""

        result = runner.invoke(
            app,
            [
                "create-generators",
            ],
            catch_exceptions=False,
        )

        mock_make.assert_called_once_with(
            mock_load_meta.return_value,
            mock_config.return_value,
            "orm.yaml",
            "config.yaml",
            None,
        )
        mock_path.return_value.write_text.assert_called_once_with(
            "some text", encoding="utf-8"
        )
        self.assertSuccess(result)

    @patch("datafaker.main.read_config_file")
    @patch("datafaker.main.load_metadata")
    @patch("datafaker.main.get_settings")
    @patch("datafaker.main.Path")
    @patch("datafaker.main.make_table_generators")
    @patch("datafaker.main.generators_require_stats")
    def test_create_generators_uses_default_stats_file_if_necessary(
        self,
        mock_require_stats: MagicMock,
        mock_make: MagicMock,
        mock_path: MagicMock,
        mock_settings: MagicMock,
        mock_load_meta: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Test the create-generators sub-command."""
        mock_require_stats.return_value = True
        mock_path.return_value.exists.return_value = False
        mock_make.return_value = "some text"
        mock_settings.return_value.src_postges_dsn = ""

        result = runner.invoke(
            app,
            [
                "create-generators",
            ],
            catch_exceptions=False,
        )

        mock_make.assert_called_once_with(
            mock_load_meta.return_value,
            mock_config.return_value,
            "orm.yaml",
            "config.yaml",
            "src-stats.yaml",
        )
        mock_path.return_value.write_text.assert_called_once_with(
            "some text", encoding="utf-8"
        )
        self.assertSuccess(result)

    @patch("datafaker.main.Path")
    @patch("datafaker.main.logger")
    def test_create_generators_errors_if_file_exists(
        self, mock_logger: MagicMock, mock_path: MagicMock
    ) -> None:
        """Test the create-generators sub-command doesn't overwrite."""

        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__str__.return_value = "df.py"

        result = runner.invoke(
            app,
            [
                "create-generators",
            ],
            catch_exceptions=False,
        )
        mock_logger.error.assert_called_once_with(
            "%s should not already exist. Exiting...", mock_path.return_value
        )
        self.assertEqual(1, result.exit_code)

    @patch("datafaker.main.read_config_file")
    @patch("datafaker.main.load_metadata")
    @patch("datafaker.main.get_settings")
    @patch("datafaker.main.Path")
    @patch("datafaker.main.make_table_generators")
    def test_create_generators_with_force_enabled(
        self,
        mock_make: MagicMock,
        mock_path: MagicMock,
        mock_settings: MagicMock,
        mock_load_meta: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Tests the create-generators sub-commands overwrite files when instructed."""

        mock_path.return_value.exists.return_value = True
        mock_make.return_value = "make result"
        mock_settings.return_value.src_postges_dsn = ""

        for force_option in ["--force", "-f"]:
            with self.subTest(f"Using option {force_option}"):
                result: Result = runner.invoke(app, [
                    "create-generators",
                    force_option,
                ])

                mock_make.assert_called_once_with(
                    mock_load_meta.return_value,
                    mock_config.return_value,
                    "orm.yaml",
                    "config.yaml",
                    None,
                )
                mock_path.return_value.write_text.assert_called_once_with(
                    "make result", encoding="utf-8"
                )
                self.assertSuccess(result)

                mock_make.reset_mock()
                mock_path.reset_mock()

    @patch("datafaker.main.create_db_tables")
    @patch("datafaker.main.read_config_file")
    @patch("datafaker.main.load_metadata")
    def test_create_tables(
        self, mock_load_meta: MagicMock, mock_config: MagicMock, mock_create: MagicMock
    ) -> None:
        """Test the create-tables sub-command."""

        result = runner.invoke(
            app,
            [
                "create-tables",
            ],
            catch_exceptions=False,
        )

        mock_create.assert_called_once_with(mock_load_meta.return_value)
        self.assertSuccess(result)

    @patch("datafaker.main.sorted_non_vocabulary_tables")
    @patch("datafaker.main.logger")
    @patch("datafaker.main.import_file")
    @patch("datafaker.main.create_db_data")
    def test_create_data(
        self,
        mock_create: MagicMock,
        mock_import: MagicMock,
        mock_logger: MagicMock,
        mock_tables: MagicMock,
    ) -> None:
        """Test the create-data sub-command."""

        mock_create.return_value = {"a": 1}
        result = runner.invoke(
            app,
            [
                "create-data",
                "--config-file=tests/examples/example_config.yaml",
                "--orm-file=tests/examples/example_orm.yaml",
            ],
            catch_exceptions=False,
        )
        self.assertListEqual([call("df.py")], mock_import.call_args_list)

        mock_create.assert_called_once_with(
            mock_tables.return_value,
            mock_import.return_value.table_generator_dict,
            mock_import.return_value.story_generator_list,
            1,
        )
        self.assertSuccess(result)

        mock_logger.debug.assert_has_calls(
            [
                call("Creating data."),
                call("Data created in %s %s.", 1, "pass"),
                call("%s: %s %s created.", "a", 1, "row"),
            ]
        )

    @patch("datafaker.main.Path")
    @patch("datafaker.main.make_tables_file")
    @patch("datafaker.main.get_settings")
    @patch("datafaker.main.read_config_file")
    def test_make_tables(
        self,
        mock_config_yaml_file: MagicMock,
        mock_get_settings: MagicMock,
        mock_make_tables_file: MagicMock,
        mock_path: MagicMock,
    ) -> None:
        """Test the make-tables sub-command."""

        mock_config = MagicMock()
        mock_path.return_value.exists.return_value = False
        mock_get_settings.return_value = get_test_settings()
        mock_make_tables_file.return_value = "some text"
        mock_config_yaml_file.return_value = mock_config

        result = runner.invoke(
            app,
            [
                "make-tables",
                "--config-file=config.yaml",
            ],
            catch_exceptions=False,
        )

        mock_make_tables_file.assert_called_once_with(
            "postgresql://suser:spassword@shost:5432/sdbname", None, mock_config
        )
        mock_path.return_value.write_text.assert_called_once_with(
            "some text", encoding="utf-8"
        )
        self.assertSuccess(result)

    @patch("datafaker.main.Path")
    @patch("datafaker.main.logger")
    def test_make_tables_errors_if_file_exists(
        self, mock_logger: MagicMock, mock_path: MagicMock
    ) -> None:
        """Test the make-tables sub-command doesn't overwrite."""

        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__str__.return_value = "orm.py"

        result = runner.invoke(
            app,
            [
                "make-tables",
            ],
            catch_exceptions=False,
        )
        mock_logger.error.assert_called_once_with(
            "%s should not already exist. Exiting...", mock_path.return_value
        )
        self.assertEqual(1, result.exit_code)

    @patch.dict(os.environ, {"SRC_SCHEMA": "myschema"}, clear=True)
    @patch("datafaker.main.logger")
    def test_make_tables_errors_if_src_dsn_missing(
        self, mock_logger: MagicMock
    ) -> None:
        """Test the make-tables sub-command refuses to work if SRC_DSN is not set."""

        result = runner.invoke(
            app,
            [
                "make-tables",
                "--orm-file=tests/examples/does-not-exist.yaml",
            ],
            catch_exceptions=False,
        )
        mock_logger.error.assert_called_once_with(
            "Missing source database connection details."
        )
        self.assertEqual(1, result.exit_code)

    @patch("datafaker.main.make_tables_file")
    @patch("datafaker.main.Path")
    @patch("datafaker.main.get_settings")
    def test_make_tables_with_force_enabled(
        self,
        mock_get_settings: MagicMock,
        mock_path: MagicMock,
        mock_make_tables: MagicMock,
    ) -> None:
        """Test the make-tables sub-command, when the force option is activated."""
        mock_get_settings.return_value = get_test_settings()
        mock_path.return_value.exists.return_value = True
        mock_tables_output: str = "make_tables_file output"

        mock_make_tables.return_value = mock_tables_output

        for force_option in ["--force", "-f"]:
            with self.subTest(f"Using option {force_option}"):
                result: Result = runner.invoke(app, [
                    "make-tables",
                    force_option,
                    "--orm-file=tests/examples/example_orm.yaml",
                ])

                mock_make_tables.assert_called_once_with(
                    mock_get_settings.return_value.src_dsn,
                    mock_get_settings.return_value.src_schema,
                    {},
                )
                mock_path.return_value.write_text.assert_called_once_with(
                    mock_tables_output, encoding="utf-8"
                )
                self.assertSuccess(result)

                mock_make_tables.reset_mock()
                mock_path.reset_mock()

    @patch("datafaker.main.Path")
    @patch("datafaker.main.make_src_stats")
    @patch("datafaker.main.get_settings")
    @patch("datafaker.main.load_metadata", side_effect=["ms"])
    def test_make_stats(
        self, _lm: MagicMock, mock_get_settings: MagicMock, mock_make: MagicMock, mock_path: MagicMock
    ) -> None:
        """Test the make-stats sub-command."""
        example_conf_path = "tests/examples/example_config.yaml"
        output_path = Path("make_stats_output.yaml")
        mock_path.return_value.exists.return_value = False
        mock_make.return_value = {"a": 1}
        mock_get_settings.return_value = get_test_settings()
        result = runner.invoke(
            app,
            [
                "make-stats",
                f"--stats-file={output_path}",
                f"--config-file={example_conf_path}",
            ],
            catch_exceptions=False,
        )
        self.assertSuccess(result)
        with open(example_conf_path, "r", encoding="utf8") as f:
            config = yaml.safe_load(f)
        mock_make.assert_called_once_with(get_test_settings().src_dsn, config, "ms", None)
        mock_path.return_value.write_text.assert_called_once_with(
            "a: 1\n", encoding="utf-8"
        )

    @patch("datafaker.main.Path")
    @patch("datafaker.main.logger")
    def test_make_stats_errors_if_file_exists(
        self, mock_logger: MagicMock, mock_path: MagicMock
    ) -> None:
        """Test the make-stats sub-command when the stats file already exists."""
        mock_path.return_value.exists.return_value = True
        example_conf_path = "tests/examples/example_config.yaml"
        output_path = "make_stats_output.yaml"
        mock_path.return_value.__str__.return_value = output_path

        result = runner.invoke(
            app,
            [
                "make-stats",
                f"--stats-file={output_path}",
                f"--config-file={example_conf_path}",
            ],
            catch_exceptions=False,
        )
        mock_logger.error.assert_called_once_with(
            "%s should not already exist. Exiting...", mock_path.return_value
        )
        self.assertEqual(1, result.exit_code)

    @patch("datafaker.main.logger")
    @patch.dict(os.environ, {"SRC_SCHEMA": "myschema"}, clear=True)
    def test_make_stats_errors_if_no_src_dsn(self, mock_logger: MagicMock) -> None:
        """Test the make-stats sub-command with missing settings."""
        example_conf_path = "tests/examples/example_config.yaml"

        result = runner.invoke(
            app,
            [
                "make-stats",
                f"--config-file={example_conf_path}",
                "--stats-file=tests/examples/does-not-exist.yaml",
            ],
            catch_exceptions=False,
        )
        mock_logger.error.assert_called_once_with(
            "Missing source database connection details."
        )
        self.assertEqual(1, result.exit_code)

    @patch("datafaker.main.Path")
    @patch("datafaker.main.make_src_stats")
    @patch("datafaker.main.get_settings")
    @patch("datafaker.main.load_metadata")
    def test_make_stats_with_force_enabled(
        self, mock_meta: MagicMock, mock_get_settings: MagicMock, mock_make: MagicMock, mock_path: MagicMock
    ) -> None:
        """Tests that the make-stats command overwrite files when instructed."""
        test_config_file: str = "tests/examples/example_config.yaml"
        with open(test_config_file, "r", encoding="utf8") as f:
            config_file_content: dict = yaml.safe_load(f)

        mock_path.return_value.exists.return_value = True
        test_settings: Settings = get_test_settings()
        mock_get_settings.return_value = test_settings
        make_test_output: dict = {"some_stat": 0}
        mock_make.return_value = make_test_output

        for force_option in ["--force", "-f"]:
            with self.subTest(f"Using option {force_option}"):
                result: Result = runner.invoke(
                    app,
                    [
                        "make-stats",
                        "--stats-file=stats_file.yaml",
                        f"--config-file={test_config_file}",
                        "--orm-file=tests/examples/example_config.yaml",
                        force_option,
                    ],
                )

                mock_make.assert_called_once_with(
                    test_settings.src_dsn, config_file_content, mock_meta.return_value, None
                )
                mock_path.return_value.write_text.assert_called_once_with(
                    "some_stat: 0\n", encoding="utf-8"
                )
                self.assertSuccess(result)

                mock_make.reset_mock()
                mock_path.reset_mock()

    def test_validate_config(self) -> None:
        """Test the validate-config sub-command."""
        result = runner.invoke(
            app,
            ["validate-config", "tests/examples/example_config.yaml"],
            catch_exceptions=False,
        )

        self.assertSuccess(result)

    def test_validate_config_invalid(self) -> None:
        """Test the validate-config sub-command."""
        result = runner.invoke(
            app,
            ["validate-config", "tests/examples/invalid_config.yaml"],
            catch_exceptions=False,
        )

        self.assertEqual(1, result.exit_code)

    @patch("datafaker.main.remove_db_data")
    @patch("datafaker.main.read_config_file")
    @patch("datafaker.main.load_metadata")
    def test_remove_data(
        self,
        mock_meta: MagicMock,
        mock_config: MagicMock,
        mock_remove: MagicMock,
    ) -> None:
        """Test the remove-data command."""
        result = runner.invoke(
            app,
            ["remove-data", "--yes"],
            catch_exceptions=False,
        )
        self.assertEqual(0, result.exit_code)
        mock_remove.assert_called_once_with(mock_meta.return_value, mock_config.return_value)

    @patch("datafaker.main.read_config_file")
    @patch("datafaker.main.remove_db_vocab")
    @patch("datafaker.main.load_metadata_config")
    @patch("datafaker.main.dict_to_metadata")
    def test_remove_vocab(
        self,
        mock_d2m: MagicMock,
        mock_load_metadata: MagicMock,
        mock_remove: MagicMock,
        mock_read_config: MagicMock,
    ) -> None:
        """Test the remove-vocab command."""
        result = runner.invoke(
            app,
            ["remove-vocab", "--yes"],
            catch_exceptions=False,
        )
        self.assertEqual(0, result.exit_code)
        mock_read_config.assert_called_once_with("config.yaml")
        mock_remove.assert_called_once_with(mock_d2m.return_value, mock_load_metadata.return_value, mock_read_config.return_value)

    @patch("datafaker.main.remove_db_tables")
    @patch("datafaker.main.load_metadata")
    @patch("datafaker.main.read_config_file")
    def test_remove_tables(self, _: MagicMock, mock_meta: MagicMock, mock_remove: MagicMock) -> None:
        """Test the remove-tables command."""
        result = runner.invoke(
            app,
            ["remove-tables", "--yes"],
            catch_exceptions=False,
        )
        self.assertEqual(0, result.exit_code)
        mock_remove.assert_called_once_with(mock_meta.return_value)
