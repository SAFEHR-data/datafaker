"""Test the parquet-file-dir to ``orm.yaml`` functionality."""
import datetime
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
from sqlalchemy import text
from typer import Exit

from datafaker.db_utils import create_db_engine, get_sync_engine
from datafaker.parquet2orm import get_parquet_orm


class HasValues:
    """Mock object that compares equal if it has the same elements."""

    def __init__(self, values: Iterable[Any]) -> None:
        """Set the values as any iterable."""
        self.value_set = set(values)

    def __eq__(self, obj: Any) -> bool:
        """Test for the correct elements."""
        return isinstance(obj, Iterable) and set(obj) == self.value_set

    def __ne__(self, obj: Any) -> bool:
        """Test for the correct elements and negate."""
        return not self.__eq__(obj)

    def __repr__(self) -> str:
        """Show the elements we want."""
        return f"HasValues<{self.value_set}>"


class ParquetDirTestCase(TestCase):
    """Base class providing a temporary directory for parquet files."""

    def setUp(self) -> None:
        """Make a temporary directory for the parquet files."""
        super().setUp()
        self.parquet_dir = Path(tempfile.mkdtemp(prefix="parq"))

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        for r, ds, fs in os.walk(self.parquet_dir, topdown=False):
            root = Path(r)
            for f in fs:
                (root / f).unlink()
            for d in ds:
                (root / d).rmdir()
        self.parquet_dir.rmdir()
        return super().tearDown()

    def write_parquet(self, data: dict[str, dict[str, list[Any]]]) -> None:
        """
        Write parquet files to the parquet directory.
        :param data: dict of file names to dict of column names to list of data.
        """
        for fn, table_data in data.items():
            pd.DataFrame.from_dict(table_data).to_parquet(self.parquet_dir / fn)


class Parquet2Orm(ParquetDirTestCase):
    """Tests the ``parquet2orm`` function."""

    def test_can_infer_column_type(self) -> None:
        """Test that we can guess obvious column types in parquet files."""
        data: dict[str, dict[str, list[Any]]] = {
            "fruit.parquet": {
                "FruitKey": [1, 2, 3],
                "orange": [True, True, False],
                "banana": ["one", "two", "three"],
                "grapes": [
                    datetime.datetime(1999, 12, 31, 23, 59, 59),
                    datetime.datetime(1999, 12, 1, 0, 0, 10),
                    datetime.datetime(2001, 6, 15, 15, 30, 16),
                ],
            }
        }
        self.write_parquet(data)
        orm = get_parquet_orm(self.parquet_dir)
        assert orm is not None
        self.assertSetEqual(set(orm.keys()), set(data.keys()))
        self.assertIn("columns", orm["fruit.parquet"])
        self.assertSetEqual(
            set(orm["fruit.parquet"]["columns"].keys()),
            set(data["fruit.parquet"].keys()),
        )
        cols = orm["fruit.parquet"]["columns"]
        self.assertIn("type", cols["FruitKey"])
        self.assertEqual(cols["FruitKey"]["type"], "INTEGER")
        self.assertIn("type", cols["orange"])
        self.assertEqual(cols["orange"]["type"], "BOOLEAN")
        self.assertIn("type", cols["banana"])
        self.assertEqual(cols["banana"]["type"], "TEXT")
        self.assertIn("type", cols["grapes"])
        self.assertEqual(cols["grapes"]["type"], "DATETIME")

    @patch("datafaker.parquet2orm.logger")
    def test_can_infer_primary_key(self, mock_logger: MagicMock) -> None:
        """Test that we can guess obvious primary keys in parquet files."""
        data: dict[str, dict[str, list[Any]]] = {
            "fruit.parquet": {
                "fruit_id": [1, 2, 3],
                "orange": [True, True, False],
                "fruit_key": ["one", "two", "three"],
                "FruitKey": [3, 2, 1],
            }
        }
        self.write_parquet(data)
        orm = get_parquet_orm(self.parquet_dir)
        assert orm is not None
        self.assertSetEqual(set(orm.keys()), set(data.keys()))
        primary_keys = {
            name
            for name, col in orm["fruit.parquet"]["columns"].items()
            if col.get("primary", False)
        }
        # not fruit_key because that column does not have integer type
        self.assertSetEqual(primary_keys, {"fruit_id", "FruitKey"})
        mock_logger.warning.assert_called_once_with(
            "Found multiple likely primary keys for table %s: %s",
            "fruit.parquet",
            HasValues({"fruit_id", "FruitKey"}),
        )

    def test_can_infer_foreign_key(self) -> None:
        """Test that we can guess obvious foreign keys in parquet files."""
        data: dict[str, dict[str, list[Any]]] = {
            "fruit.parquet": {
                "fruit_id": [1, 2],
                "name": ["grape", "orange"],
                "seed_id": [2, 1],
            },
            "seed.parquet": {"seed_id": [1, 2], "name": ["orange pip", "grape seed"]},
        }
        self.write_parquet(data)
        orm = get_parquet_orm(self.parquet_dir)
        assert orm is not None
        self.assertSetEqual(set(orm.keys()), set(data.keys()))
        self.assertNotIn("foreign_keys", orm["fruit.parquet"]["columns"]["fruit_id"])
        self.assertNotIn("foreign_keys", orm["fruit.parquet"]["columns"]["name"])
        self.assertNotIn("foreign_keys", orm["seed.parquet"]["columns"]["seed_id"])
        self.assertNotIn("foreign_keys", orm["seed.parquet"]["columns"]["name"])
        self.assertIn("foreign_keys", orm["fruit.parquet"]["columns"]["seed_id"])
        self.assertListEqual(
            orm["fruit.parquet"]["columns"]["seed_id"]["foreign_keys"],
            ["seed.parquet.seed_id"],
        )


class PartitionedParquet2Orm(ParquetDirTestCase):
    """Tests reading directories of parquet files as single tables."""

    def write_files(self, data: dict[str, dict[str, list[Any]]]) -> None:
        """
        Write parquet files at the relative paths given.

        :param data: dict of relative file paths to dict of column names to data.
        """
        for fn, table_data in data.items():
            path = self.parquet_dir / fn
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_dict(table_data).to_parquet(path)

    def test_partitioned_directory_is_one_table(self) -> None:
        """Test a hive-partitioned directory is one table named after it."""
        self.write_files(
            {
                "visit/yr=2023/p00001.parquet": {"visit_id": [1], "ward": ["a"]},
                "visit/yr=2023/p00002.parquet": {"visit_id": [2], "ward": ["b"]},
                "visit/yr=__MISSING__/p00001.parquet": {"visit_id": [3], "ward": ["c"]},
            }
        )
        orm = get_parquet_orm(self.parquet_dir)
        assert orm is not None
        self.assertSetEqual(set(orm.keys()), {"visit"})
        cols = orm["visit"]["columns"]
        self.assertSetEqual(set(cols.keys()), {"visit_id", "ward", "yr"})
        self.assertEqual(cols["yr"]["type"], "TEXT")
        self.assertEqual(cols["visit_id"]["type"], "INTEGER")

    @patch("datafaker.parquet2orm.logger")
    def test_partition_key_type_is_not_a_warning(self, mock_logger: MagicMock) -> None:
        """Test the partition key does not log 'Could not determine type'."""
        self.write_files({"visit/yr=2023/p1.parquet": {"visit_id": [1]}})
        get_parquet_orm(self.parquet_dir)
        mock_logger.warning.assert_not_called()

    def test_same_part_names_in_different_tables_do_not_collide(self) -> None:
        """Test part files of the same name in different datasets stay separate."""
        self.write_files(
            {
                "visit/yr=2023/p00001.parquet": {"visit_id": [1]},
                "drug/yr=2023/p00001.parquet": {"drug_id": [1], "dose": [1.5]},
            }
        )
        orm = get_parquet_orm(self.parquet_dir)
        assert orm is not None
        self.assertSetEqual(set(orm.keys()), {"visit", "drug"})
        self.assertSetEqual(
            set(orm["drug"]["columns"].keys()), {"drug_id", "dose", "yr"}
        )
        self.assertSetEqual(set(orm["visit"]["columns"].keys()), {"visit_id", "yr"})

    def test_files_and_directories_mix(self) -> None:
        """Test a mixture of loose parquet files and parquet dirs are read as separate tables.
        Checks empty directory are not treated as partitioned tables."""
        self.write_files(
            {
                "fruit.parquet": {"fruit_id": [1]},
                "visit/yr=2023/p1.parquet": {"visit_id": [1]},
            }
        )
        (self.parquet_dir / "empty").mkdir()

        orm = get_parquet_orm(self.parquet_dir)
        assert orm is not None
        self.assertSetEqual(set(orm.keys()), {"fruit.parquet", "visit"})

    def test_nested_plain_directories_are_one_table(self) -> None:
        """Test a subdirectory of unpartitioned files is one table."""
        self.write_files(
            {
                "sub/a.parquet": {"x": [1]},
                "sub/b.parquet": {"x": [2]},
            }
        )
        orm = get_parquet_orm(self.parquet_dir)
        assert orm is not None
        self.assertSetEqual(set(orm.keys()), {"sub"})

    def test_parquet_dir_that_is_a_dataset(self) -> None:
        """Test the directory given can itself be a partitioned table."""
        self.write_files(
            {
                "yr=2022/p1.parquet": {"visit_id": [1]},
                "yr=2023/p1.parquet": {"visit_id": [2]},
            }
        )
        orm = get_parquet_orm(self.parquet_dir)
        assert orm is not None
        self.assertSetEqual(set(orm.keys()), {self.parquet_dir.resolve().name})
        only = next(iter(orm.values()))
        self.assertSetEqual(set(only["columns"].keys()), {"visit_id", "yr"})

    def test_parquet_dir_dataset_must_be_consistent(self) -> None:
        """Test a directory of unrelated hive-named tables is still rejected."""
        self.write_files(
            {
                "yr=2022/p1.parquet": {"visit_id": [1]},
                "yr=2023/p1.parquet": {"other": ["x"]},
            }
        )
        with self.assertRaises(Exit):
            get_parquet_orm(self.parquet_dir)

    @patch("datafaker.parquet2orm.logger")
    def test_differing_columns_are_rejected(self, mock_logger: MagicMock) -> None:
        """Test different tables nested under one directory give a clear error."""
        self.write_files(
            {
                "sub/a.parquet": {"visit_id": [1], "ward": ["a"]},
                "sub/b.parquet": {"drug_id": [1]},
            }
        )
        with self.assertRaises(Exit):
            get_parquet_orm(self.parquet_dir)
        mock_logger.error.assert_called_once()
        message = (
            mock_logger.error.call_args.args[0] % mock_logger.error.call_args.args[1:]
        )
        self.assertIn("sub", message)
        self.assertIn("missing columns ['visit_id', 'ward']", message)
        self.assertIn("extra columns ['drug_id']", message)
        self.assertIn("move each table into its own directory", message)

    @patch("datafaker.parquet2orm.logger")
    def test_differing_types_are_a_warning(self, mock_logger: MagicMock) -> None:
        """Test the same column with a different type warns but is accepted."""
        self.write_files(
            {
                "visit/yr=2022/p1.parquet": {"visit_id": [1]},
                "visit/yr=2023/p1.parquet": {"visit_id": [1.5]},
                "visit/yr=2023/p2.parquet": {"visit_id": [2.5]},
            }
        )
        orm = get_parquet_orm(self.parquet_dir)
        assert orm is not None
        self.assertEqual(orm["visit"]["columns"]["visit_id"]["type"], "INTEGER")
        mock_logger.error.assert_not_called()
        mock_logger.warning.assert_called_once()
        args = mock_logger.warning.call_args.args
        self.assertEqual(args[1:3], ("visit", "visit_id"))
        self.assertEqual(args[3], ["float64", "int64"])

    @patch("datafaker.parquet2orm.logger")
    def test_int32_and_nullable_int32_are_the_same_type(
        self, mock_logger: MagicMock
    ) -> None:
        """Test ``int32`` and ``Int32`` do not even warn."""
        self.write_files(
            {
                "person/yr=2022/p1.parquet": {"age": [1]},
                "person/yr=2023/p1.parquet": {"age": [2]},
            }
        )
        pd.DataFrame({"age": pd.array([1], dtype="int32")}).to_parquet(
            self.parquet_dir / "person/yr=2022/p1.parquet"
        )
        pd.DataFrame({"age": pd.array([None, 2], dtype="Int32")}).to_parquet(
            self.parquet_dir / "person/yr=2023/p1.parquet"
        )
        orm = get_parquet_orm(self.parquet_dir)
        assert orm is not None
        self.assertEqual(orm["person"]["columns"]["age"]["type"], "INTEGER")
        for call in mock_logger.warning.call_args_list:
            self.assertNotIn("different types", call.args[0])

    def test_nullable_int_first_file_is_still_integer(self) -> None:
        """Test a pandas ``Int32`` column is typed ``INTEGER``."""
        self.write_files({"person/yr=2022/p1.parquet": {"age": [1]}})
        pd.DataFrame({"age": pd.array([None, 2], dtype="Int32")}).to_parquet(
            self.parquet_dir / "person/yr=2022/p1.parquet"
        )
        orm = get_parquet_orm(self.parquet_dir)
        assert orm is not None
        self.assertEqual(orm["person"]["columns"]["age"]["type"], "INTEGER")

    def test_differing_partition_keys_are_rejected(self) -> None:
        """Test files partitioned differently are rejected."""
        self.write_files(
            {
                "visit/yr=2022/p1.parquet": {"visit_id": [1]},
                "visit/p2.parquet": {"visit_id": [2]},
            }
        )
        with self.assertRaises(Exit):
            get_parquet_orm(self.parquet_dir)


class PartitionedParquetEngine(TestCase):
    """Integration tests for querying Parquet data through the parquet_dir option."""

    def setUp(self) -> None:
        """Create partitioned and loose Parquet data for parquet_dir integration tests."""
        super().setUp()
        self.parquet_dir = Path(tempfile.mkdtemp(prefix="parq"))
        self.addCleanup(shutil.rmtree, self.parquet_dir)
        for path, data in {
            "visit/yr=2022/p1.parquet": {"visit_id": [1, 2]},
            "visit/yr=2023/p1.parquet": {"visit_id": [3]},
            "visit/yr=2023/p2.parquet": {"visit_id": [4]},
            "fruit.parquet": {"fruit_id": [9]},
        }.items():
            (self.parquet_dir / path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_dict(data).to_parquet(self.parquet_dir / path)

    def test_dataset_can_be_queried_by_directory_name(self) -> None:
        """Test every partition appears in the table named after the directory."""
        engine = get_sync_engine(
            create_db_engine("duckdb:///:memory:", parquet_dir=self.parquet_dir)
        )
        with engine.connect() as conn:
            rows = conn.execute(
                text('SELECT a.visit_id, a.yr FROM "visit" AS a ORDER BY a.visit_id')
            ).fetchall()
            fruit = conn.execute(text('SELECT * FROM "fruit.parquet"')).fetchall()
        self.assertListEqual(
            [tuple(r) for r in rows], [(1, 2022), (2, 2022), (3, 2023), (4, 2023)]
        )
        self.assertListEqual([tuple(r) for r in fruit], [(9,)])

    def test_dataset_views_are_not_written_to_the_source_database(self) -> None:
        """Test a file-backed source database is not modified."""
        db = self.parquet_dir / "src.duckdb"
        engine = get_sync_engine(
            create_db_engine(f"duckdb:///{db}", parquet_dir=self.parquet_dir)
        )
        with engine.connect() as conn:
            conn.execute(text('SELECT count(*) FROM "visit"'))
        engine.dispose()
        engine = get_sync_engine(create_db_engine(f"duckdb:///{db}"))
        with engine.connect() as conn:
            with self.assertRaises(Exception):
                conn.execute(text('SELECT count(*) FROM "visit"'))

    def test_parquet_dir_may_be_a_string(self) -> None:
        """Test the directory read back from ``orm.yaml`` as text still works."""
        engine = get_sync_engine(
            create_db_engine("duckdb:///:memory:", parquet_dir=str(self.parquet_dir))
        )
        with engine.connect() as conn:
            count = conn.execute(text('SELECT count(*) FROM "visit"')).scalar()
        self.assertEqual(count, 4)
