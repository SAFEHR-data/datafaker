"""Test the parquet-file-dir to ``orm.yaml`` functionality."""
import datetime
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

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


class Parquet2Orm(TestCase):
    """Tests the ``parquet2orm`` function."""

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
