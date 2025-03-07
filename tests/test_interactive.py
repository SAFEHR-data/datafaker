"""Tests for the base module."""
from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

from sqlsynthgen.interactive import TableCmd
from tests.utils import RequiresDBTestCase


class TestTableCmd(TableCmd):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()
    def reset(self):
        self.messages = []
        self.headings = []
        self.rows = []
    def print(self, text: str, *args, **kwargs):
        self.messages.append((text, args, kwargs))
    def print_table(self, headings: list[str], rows: list[list[str]]):
        self.headings = headings
        self.rows = rows
    def ask_save(self):
        return "yes"


class ConfigureTablesTests(RequiresDBTestCase):
    """Testing configure-tables."""
    dump_file_path = "src.dump"
    database_name = "src"
    #schema_name = "public"

    def test_table_name_prompts(self) -> None:
        """Test that the prompts follow the names of the tables."""
        metadata = MetaData()
        metadata.reflect(self.engine)
        config = {}
        with TestTableCmd(self.dsn, self.schema_name, metadata, config) as tc:
            table_names = list(metadata.tables.keys())
            for t in table_names:
                self.assertIn(t, tc.prompt)
                tc.do_next("")
            self.assertListEqual(tc.messages, [(TableCmd.ERROR_NO_MORE_TABLES, (), {})])
            tc.reset()
            for t in reversed(table_names):
                self.assertIn(t, tc.prompt)
                tc.do_previous("")
            self.assertListEqual(tc.messages, [(TableCmd.ERROR_ALREADY_AT_START, (), {})])
