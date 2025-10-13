"""Utilities for testing."""
import asyncio
import os
import shutil
import traceback
from collections.abc import MutableSequence, Sequence
from functools import lru_cache
from pathlib import Path
from subprocess import run
from tempfile import mkstemp
from typing import Any, Mapping
from unittest import TestCase, skipUnless

import testing.postgresql
import yaml
from sqlalchemy.schema import MetaData

from datafaker import settings
from datafaker.create import create_db_data_into
from datafaker.interactive.base import DbCmd
from datafaker.make import make_src_stats, make_table_generators, make_tables_file
from datafaker.remove import remove_db_data_from
from datafaker.utils import (
    T,
    create_db_engine,
    get_sync_engine,
    import_file,
    sorted_non_vocabulary_tables,
)


class SysExit(Exception):
    """To force the function to exit as sys.exit() would."""


@lru_cache(1)
def get_test_settings() -> settings.Settings:
    """Get a Settings object that ignores .env files and environment variables."""

    return settings.Settings(
        src_dsn="postgresql://suser:spassword@shost:5432/sdbname",
        dst_dsn="postgresql://duser:dpassword@dhost:5432/ddbname",
        # To stop any local .env files influencing the test
        # The mypy ignore can be removed once we upgrade to pydantic 2.
        _env_file=None,  # type: ignore[call-arg]
    )


class DatafakerTestCase(TestCase):
    """Parent class for all TestCases in datafaker."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize an instance of DatafakerTestCase."""
        self.maxDiff = None  # pylint: disable=invalid-name
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        settings.get_settings.cache_clear()

    def assertReturnCode(  # pylint: disable=invalid-name
        self, result: Any, expected_code: int
    ) -> None:
        """Give details for a subprocess result and raise if it's not as expected."""
        code = result.exit_code if hasattr(result, "exit_code") else result.returncode
        if code != expected_code:
            print(result.stdout)
            print(result.stderr)
            self.assertEqual(expected_code, code)

    def assertSuccess(self, result: Any) -> None:  # pylint: disable=invalid-name
        """Give details for a subprocess result and raise if the result isn't good."""
        self.assertReturnCode(result, 0)

    def assertFailure(self, result: Any) -> None:  # pylint: disable=invalid-name
        """Give details for a subprocess result and raise if the result isn't bad."""
        self.assertReturnCode(result, 1)

    def assertNoException(self, result: Any) -> None:  # pylint: disable=invalid-name
        """Assert that the result has no exception."""
        if result.exception is None:
            return
        self.fail("".join(traceback.format_exception(result.exception)))

    def assert_greater_and_not_none(self, left: float | None, right: float) -> None:
        """
        Assert left is not None and greater than right
        """
        if left is None:
            self.fail("first argument is None")
        else:
            self.assertGreater(left, right)

    def assert_subset(self, set1: set[T], set2: set[T], msg: str | None = None) -> None:
        """Assert a set is a (non-strict) subset.

        :param set1: The asserted subset.
        :param set2: The asserted superset.
        :param msg: Optional message to use on failure instead of a list of
        differences.
        """
        try:
            difference = set1.difference(set2)
        except TypeError as e:
            self.fail(f"invalid type when attempting set difference: {e}")
        except AttributeError as e:
            self.fail(f"first argument does not support set difference: {e}")

        if not difference:
            return

        lines = []
        if difference:
            lines.append("Items in the first set but not the second:")
            for item in difference:
                lines.append(repr(item))

        standard_msg = "\n".join(lines)
        self.fail(self._formatMessage(msg, standard_msg))


@skipUnless(shutil.which("psql"), "need to find 'psql': install PostgreSQL to enable")
class RequiresDBTestCase(DatafakerTestCase):
    """
    A test case that only runs if PostgreSQL is installed.
    A test postgres is installed
    dump_file_path can be set to run in this postgres database.
    database_name is the name of the database referred to in dump_file_path.
    You can use ``self.dsn`` to retrieve the DSN of this database, ``self.engine``
    to get an engine to access the database and self.metadata to get metadata
    reflected from that engine.
    """

    schema_name: str | None = None
    use_asyncio = False
    examples_dir = Path("tests/examples")
    dump_file_path: str | None = None
    database_name: str | None = None
    Postgresql = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.Postgresql = testing.postgresql.PostgresqlFactory(cache_initialized_db=True)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.Postgresql is not None:
            cls.Postgresql.clear_cache()

    def setUp(self) -> None:
        super().setUp()
        assert self.Postgresql is not None
        self.postgresql = self.Postgresql()  # pylint: disable=not-callable
        if self.dump_file_path is not None:
            self.run_psql(Path(self.examples_dir) / Path(self.dump_file_path))
        self.engine = create_db_engine(
            self.dsn,
            schema_name=self.schema_name,
            use_asyncio=self.use_asyncio,
        )
        self.sync_engine = get_sync_engine(self.engine)
        self.metadata = MetaData()
        self.metadata.reflect(self.sync_engine)

    def tearDown(self) -> None:
        self.postgresql.stop()
        super().tearDown()

    @property
    def dsn(self) -> str:
        """Get the database connection string."""
        if self.database_name:
            url = self.postgresql.url(database=self.database_name)
        else:
            url = self.postgresql.url()
        assert isinstance(url, str)
        return url

    def run_psql(self, dump_file: Path) -> None:
        """Run psql and pass dump_file_name as the --file option."""

        # If you need to update a .dump file, use
        # PGPASSWORD=password pg_dump \
        # --host=localhost \
        # --port=5432 \
        # --dbname=src \
        # --username=postgres \
        # --no-password \
        # --clean \
        # --create \
        # --insert \
        # --if-exists > tests/examples/FILENAME.dump

        # Clear and re-create the test database
        completed_process = run(
            ["psql", "-d", self.postgresql.url(), "-f", dump_file],
            capture_output=True,
            check=True,
        )
        # psql doesn't always return != 0 if it fails
        assert completed_process.stderr == b"", completed_process.stderr


class GeneratesDBTestCase(RequiresDBTestCase):
    """A test case for which a database is generated."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise a GeneratedDB test case."""
        super().__init__(*args, **kwargs)
        self.generators_file_path = ""
        self.stats_fd = 0
        self.stats_file_path = ""
        self.config_file_path = ""
        self.config_fd = 0

    def setUp(self) -> None:
        """Set up the test case with an actual orm.yaml file."""
        super().setUp()
        # Generate the `orm.yaml` from the database
        (self.orm_fd, self.orm_file_path) = mkstemp(".yaml", "orm_", text=True)
        with os.fdopen(self.orm_fd, "w", encoding="utf-8") as orm_fh:
            orm_fh.write(make_tables_file(self.dsn, self.schema_name, {}))

    def set_configuration(self, config: Mapping[str, Any]) -> None:
        """Accepts a configuration file, writes it out."""
        (self.config_fd, self.config_file_path) = mkstemp(".yaml", "config_", text=True)
        with os.fdopen(self.config_fd, "w", encoding="utf-8") as config_fh:
            config_fh.write(yaml.dump(config))

    def get_src_stats(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """
        Runs `make-stats` producing `src-stats.yaml`.

        :return: Python dictionary representation of the contents of the src-stats file
        """
        loop = asyncio.new_event_loop()
        src_stats = loop.run_until_complete(
            make_src_stats(self.dsn, config, self.schema_name)
        )
        loop.close()
        (self.stats_fd, self.stats_file_path) = mkstemp(
            ".yaml", "src_stats_", text=True
        )
        with os.fdopen(self.stats_fd, "w", encoding="utf-8") as stats_fh:
            stats_fh.write(yaml.dump(src_stats))
        return src_stats

    def create_generators(self, config: Mapping[str, Any]) -> None:
        """``create-generators`` with ``src-stats.yaml`` and the rest, producing ``df.py``"""
        datafaker_content = make_table_generators(
            self.metadata,
            config,
            self.orm_file_path,
            self.config_file_path,
            self.stats_file_path,
        )
        (generators_fd, self.generators_file_path) = mkstemp(".py", "dfgen_", text=True)
        with os.fdopen(generators_fd, "w", encoding="utf-8") as datafaker_fh:
            datafaker_fh.write(datafaker_content)

    def remove_data(self, config: Mapping[str, Any]) -> None:
        """Remove source data from the DB."""
        # `remove-data` so we don't have to use a separate database for the destination
        remove_db_data_from(self.metadata, config, self.dsn, self.schema_name)

    def create_data(self, config: Mapping[str, Any], num_passes: int = 1) -> None:
        """Create fake data in the DB."""
        # `create-data` with all this stuff
        datafaker_module = import_file(self.generators_file_path)
        create_db_data_into(
            sorted_non_vocabulary_tables(self.metadata, config),
            datafaker_module,
            num_passes,
            self.dsn,
            self.schema_name,
        )

    def generate_data(
        self, config: Mapping[str, Any], num_passes: int = 1
    ) -> Mapping[str, Any]:
        """
        Replaces the DB's source data with generated data.
        :return: A Python dictionary representation of the src-stats.yaml file, for what it's worth.
        """
        self.set_configuration(config)
        src_stats = self.get_src_stats(config)
        self.create_generators(config)
        self.remove_data(config)
        self.create_data(config, num_passes)
        return src_stats


class TestDbCmdMixin(DbCmd):
    """A mixin for capturing output from interactive commands."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a TestDbCmdMixin"""
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self) -> None:
        """Reset all the debug messages collected so far."""
        self.messages: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.headings: Sequence[str] = []
        self.rows: Sequence[Sequence[str]] = []
        self.column_items: MutableSequence[Sequence[str]] = []
        self.columns: Mapping[str, Sequence[Any]] = {}

    def print(self, text: str, *args: Any, **kwargs: Any) -> None:
        """Capture the printed message."""
        self.messages.append((text, args, kwargs))

    def print_table(
        self, headings: Sequence[str], rows: Sequence[Sequence[str]]
    ) -> None:
        """Capture the printed table."""
        self.headings = headings
        self.rows = rows

    def print_table_by_columns(self, columns: Mapping[str, Sequence[str]]) -> None:
        """Capture the printed table."""
        self.columns = columns

    # pylint: disable=arguments-renamed
    def columnize(self, items: Sequence[str] | None, _displaywidth: int = 80) -> None:
        """Capture the printed table."""
        if items is not None:
            self.column_items.append(items)

    def ask_save(self) -> str:
        """Quitting always works without needing to ask the user."""
        return "yes"
