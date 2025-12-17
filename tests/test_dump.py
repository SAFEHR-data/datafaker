"""Tests for the base module."""
import csv
from pathlib import Path
import tempfile

from datafaker.dump import CsvTableWriter, get_parquet_table_writer
import pandas as pd
from tests.utils import RequiresDBTestCase


class DumpTests(RequiresDBTestCase):
    """Testing configure-tables."""

    dump_file_path = "instrument.sql"
    database_name = "instrument"
    schema_name = "public"

    def test_dump_data_csv(self) -> None:
        """Test dump-data for CSV output."""
        outdir = Path(tempfile.mkdtemp("dump"))
        writer = CsvTableWriter(self.metadata, self.dsn, None)
        table_name = "manufacturer"
        writer.write(self.metadata.tables[table_name], outdir)
        table_path = outdir / f"{table_name}.csv"
        self.assertTrue(table_path.is_file())
        with table_path.open() as table_fh:
            reader = csv.reader(table_fh)
            content = list(reader)
            self.assertListEqual(content[0], ["id", "name", "founded"])
            self.assertListEqual(content[1], ["1", "Blender", "1951-01-08 12:05:06+00:00"])
            self.assertListEqual(content[2], ["2", "Gibbs", "1959-03-04 15:08:09+00:00"])

    def test_dump_data_parquet(self) -> None:
        """Test dump-data for Parquet output."""
        outdir = Path(tempfile.mkdtemp("dump"))
        writer = get_parquet_table_writer(self.metadata, self.dsn, None)
        table_name = "manufacturer"
        writer.write(self.metadata.tables[table_name], outdir)
        table_path = outdir / f"{table_name}.parquet"
        self.assertTrue(table_path.is_file())
        df = pd.read_parquet(table_path)
        self.assertListEqual(df.columns.to_list(), ["id", "name", "founded"])
        self.assertListEqual(df["id"].to_list(), [1, 2])
        self.assertListEqual(df["name"].to_list(), ["Blender", "Gibbs"])
        founded = [
            pd.Timestamp("1951-01-08 12:05:06+00:00"),
            pd.Timestamp("1959-03-04 15:08:09+00:00"),
        ]
        self.assertListEqual(df["founded"].to_list(), founded)
