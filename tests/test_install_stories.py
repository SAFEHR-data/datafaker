"""Tests for installing stories into ``config.yaml``."""
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml
from sqlalchemy import Row, func, select, text
from typer.testing import CliRunner, Result

from datafaker.main import app, install_stories
from tests.utils import GeneratesDBTestCase, create_db_engine, get_sync_engine

# pylint: disable=subprocess-run-check


class InstallTestCase(GeneratesDBTestCase):
    """End-to-end tests that require a database."""

    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    examples_dir = Path("tests/examples")

    orm_file_path = Path("orm.yaml")

    input_file_paths = [Path("annotated_stories.py"), Path("install_config.yaml")]
    stats_file_path = Path("example_stats.yaml")

    src_stats_re = re.compile(r'SRC_STATS\["(.*)"\]\["results"\]')

    start_dir = os.getcwd()

    def setUp(self) -> None:
        """Pre-test setup."""
        super().setUp()
        self.env = {
            "src_dsn": self.dsn,
            "src_schema": self.schema_name,
            "dst_dsn": self.dsn,
            "dst_schema": "dstschema",
        }
        self.runner = CliRunner(
            mix_stderr=False,
            env=self.env,
        )

        # Copy some of the example files over to the workspace.
        self.test_dir = Path(tempfile.mkdtemp(prefix="df-"))
        for file in self.input_file_paths:
            src = self.examples_dir / file
            dst = self.test_dir / file
            shutil.copy(src, dst)

        os.chdir(self.test_dir)

    def tearDown(self) -> None:
        """Tear down post test."""
        os.chdir(self.start_dir)
        super().tearDown()

    def assert_silent_success(self, completed_process: Result) -> None:
        """Assert that the process completed successfully without producing output."""
        self.assertNoException(completed_process)
        self.assertSuccess(completed_process)
        self.assertEqual(completed_process.stderr, "")
        self.assertEqual(completed_process.stdout, "")

    def test_install_stories_simple(self) -> None:
        """Test story gets expected parameters after installation."""
        config_path = Path("config-iss.yaml")
        config_path.write_text("{}", encoding="UTF-8")

        install_stories(config_path, Path("annotated_stories.py"))

        config = yaml.load(
            config_path.read_text(encoding="UTF-8"),
            Loader=yaml.SafeLoader,
        )

        # Module name configured
        self.assertIn("story_generators_module", config)
        self.assertEqual(config["story_generators_module"], "annotated_stories")

        # Generator added with parameter
        self.assertIn("story_generators", config)
        st_gen = config["story_generators"]
        self.assertEqual(len(st_gen), 1)
        self.assertIn("name", st_gen[0])
        self.assertEqual(st_gen[0]["name"], "annotated_stories.string_story_one_sd")
        self.assertIn("kwargs", st_gen[0])
        self.assertIn("stats", st_gen[0]["kwargs"])
        stats_ref = st_gen[0]["kwargs"]["stats"]
        stats_result = self.src_stats_re.match(stats_ref)
        self.assertIsNotNone(
            stats_result, f'parameter "{stats_ref}" is not a SRC_STATS reference'
        )
        assert stats_result is not None

        # Source stats query
        self.assertIn("src-stats", config)
        src_stats = config["src-stats"]
        assert src_stats is not None
        self.assertEqual(len(src_stats), 1)
        assert src_stats[0] is not None
        self.assertIn("name", src_stats[0])
        self.assertEqual(src_stats[0]["name"], stats_result.group(1))
        self.assertIn("query", src_stats[0])
        query = src_stats[0]["query"]
        (mean, stddev, _count) = self.get_string_stats()

        # Let's run the query and see what we get.
        engine = get_sync_engine(
            create_db_engine(
                self.env["src_dsn"],
                schema_name=self.env["src_schema"],
            )
        )
        with engine.connect() as conn:
            rows = conn.execute(text(query)).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].freq_mean, mean)
            self.assertEqual(rows[0].freq_stddev, stddev)

    def test_install_stories_end_to_end(self) -> None:
        """Test the stories run with the expected parameters after installation."""
        completed_process = self.invoke(
            "make-tables",
            "--force",
        )
        self.assert_silent_success(completed_process)

        completed_process = self.invoke(
            "install-stories",
            "--config-file",
            "install_config.yaml",
            "annotated_stories.py",
        )
        self.assert_silent_success(completed_process)

        completed_process = self.invoke(
            "make-stats",
            "--config-file",
            "install_config.yaml",
            "--force",
        )
        self.assert_silent_success(completed_process)

        completed_process = self.invoke(
            "create-generators",
            "--config-file",
            "install_config.yaml",
            "--force",
            "--stats-file=src-stats.yaml",
        )
        self.assert_silent_success(completed_process)

        completed_process = self.invoke(
            "remove-tables",
            "--config-file",
            "install_config.yaml",
            "--yes",
        )
        self.assert_silent_success(completed_process)

        completed_process = self.invoke(
            "create-tables",
            "--config-file",
            "install_config.yaml",
        )
        self.assert_silent_success(completed_process)

        completed_process = self.invoke(
            "create-data",
            "--config-file",
            "install_config.yaml",
        )
        self.assertNoException(completed_process)
        self.assertEqual("", completed_process.stderr)
        self.assertSuccess(completed_process)
        self.assertEqual(
            "Generating data for story 'annotated_stories.string_story_one_sd'\n",
            completed_process.stdout,
        )

        (mean, stddev, count) = self.get_string_stats()

        model_table = self.metadata.tables["model"]
        string_table = self.metadata.tables["string"]
        engine = get_sync_engine(
            create_db_engine(
                self.env["dst_dsn"],
                schema_name=self.env["dst_schema"],
            )
        )
        with engine.connect() as conn:
            row = conn.execute(
                select(model_table.c.name, model_table.c.id).where(
                    model_table.c.name == "one_sd"
                )
            ).fetchone()
            assert row is not None
            strs = conn.execute(
                select(string_table).where(string_table.c.model_id == row.id)
            ).fetchall()
            lower = None
            higher = None
            for s in strs:
                if s.position == 0:
                    self.assertIsNone(
                        lower, "Multiple one_sd strings with zero position"
                    )
                    lower = s.frequency
                else:
                    self.assertIsNone(
                        higher, "Multiple one_sd strings with non-zero position"
                    )
                    self.assertEqual(s.position, count)
                    higher = s.frequency
            assert lower is not None
            assert higher is not None
            self.assertAlmostEqual((higher + lower) / 2, mean)
            self.assertAlmostEqual((higher - lower) / 2, stddev)

    def get_string_stats(self) -> tuple[float | None, float | None, int | None]:
        """Get the mean, standard deviation and count of frequencies in the string table."""
        string_table = self.metadata.tables["string"]
        engine = get_sync_engine(
            create_db_engine(
                self.env["src_dsn"],
                schema_name=self.env["src_schema"],
            )
        )
        with engine.connect() as conn:
            results = conn.execute(
                select(
                    func.count(),  # pylint: disable=not-callable
                    func.avg(string_table.c.frequency),
                    func.stddev(string_table.c.frequency),
                )
            ).fetchone()
            if not isinstance(results, Row):
                return None, None, None
            return results.avg_1, results.stddev_1, results.count_1

    def invoke(
        self,
        *args: Any,
        expected_error: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Result:
        """
        Run datafaker with the given arguments and environment.

        :param args: Arguments to provide to datafaker.
        :param expected_error: If None, will assert that the invocation
        passes successfully without throwing an exception. Otherwise,
        the suggested error must be present in the standard error stream.
        :param env: The environment variables to be set during invocation.
        """
        res = self.runner.invoke(app, args, env=env)
        if expected_error is None:
            self.assertNoException(res)
            self.assertSuccess(res)
        else:
            self.assertIn(expected_error, res.stderr)
        return res
