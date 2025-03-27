"""Tests for the base module."""
from sqlalchemy import MetaData, select
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
        self.column_items = []
    def print(self, text: str, *args, **kwargs):
        self.messages.append((text, args, kwargs))
    def print_table(self, headings: list[str], rows: list[list[str]]):
        self.headings = headings
        self.rows = rows
    def columnize(self, items):
        self.column_items.append(items)
    def ask_save(self):
        return "yes"


class ConfigureTablesTests(RequiresDBTestCase):
    """Testing configure-tables."""
    dump_file_path = "src.dump"
    database_name = "src"
    schema_name = "public"

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
            tc.reset()
            bad_table_name = "notarealtable"
            tc.do_next(bad_table_name)
            self.assertListEqual(tc.messages, [(TableCmd.ERROR_NO_SUCH_TABLE, (bad_table_name,), {})])
            tc.reset()
            good_table_name = table_names[2]
            tc.do_next(good_table_name)
            self.assertListEqual(tc.messages, [])
            self.assertIn(good_table_name, tc.prompt)

    def test_column_display(self) -> None:
        """Test that we can see the names of the columns."""
        metadata = MetaData()
        metadata.reflect(self.engine)
        config = {}
        with TestTableCmd(self.dsn, self.schema_name, metadata, config) as tc:
            tc.do_next("unique_constraint_test")
            tc.do_columns("")
            self.assertListEqual(tc.column_items, [["id", "a", "b", "c"]])

    def test_configure_tables(self) -> None:
        """Test that we can change columns to ignore, vocab or reset."""
        metadata = MetaData()
        metadata.reflect(self.engine)
        config = {
            "tables": {
                "unique_constraint_test": {
                    "vocabulary_table": True,
                },
                "no_pk_test": {
                    "ignore": True,
                },
            },
        }
        with TestTableCmd(self.dsn, self.schema_name, metadata, config) as tc:
            tc.do_next("unique_constraint_test")
            tc.do_reset("")
            tc.do_next("person")
            tc.do_vocabulary("")
            tc.do_next("mitigation_type")
            tc.do_ignore("")
            tc.do_quit("")
            tables = tc.config["tables"]
            self.assertFalse(tables["unique_constraint_test"].get("vocabulary_table", False))
            self.assertFalse(tables["unique_constraint_test"].get("ignore", False))
            self.assertFalse(tables["no_pk_test"].get("vocabulary_table", False))
            self.assertTrue(tables["no_pk_test"].get("ignore", False))
            self.assertTrue(tables["person"].get("vocabulary_table", False))
            self.assertFalse(tables["person"].get("ignore", False))
            self.assertFalse(tables["mitigation_type"].get("vocabulary_table", False))
            self.assertTrue(tables["mitigation_type"].get("ignore", False))

    def test_print_data(self) -> None:
        """Test that we can print random rows from the table and random data from columns."""
        metadata = MetaData()
        metadata.reflect(self.engine)
        person_table = metadata.tables["person"]
        with self.engine.connect() as conn:
            person_rows = conn.execute(select(person_table)).mappings().fetchall()
            person_data = {
                row["person_id"]: row
                for row in person_rows
            }
            name_set = {row["name"] for row in person_rows}
        person_headings = ["person_id", "name", "research_opt_out", "stored_from"]
        config = {}
        with TestTableCmd(self.dsn, self.schema_name, metadata, config) as tc:
            tc.do_next("person")
            tc.do_data("")
            self.assertListEqual(tc.headings, person_headings)
            self.assertEqual(len(tc.rows), 10)  # default number of rows is 10
            for row in tc.rows:
                expected = person_data[row[0]]
                self.assertListEqual(row, [expected[h] for h in person_headings])
            tc.reset()
            rows_to_get_count = 6
            tc.do_data(str(rows_to_get_count))
            self.assertListEqual(tc.headings, person_headings)
            self.assertEqual(len(tc.rows), rows_to_get_count)
            for row in tc.rows:
                expected = person_data[row[0]]
                self.assertListEqual(row, [expected[h] for h in person_headings])
            tc.reset()
            to_get_count = 12
            tc.do_data(f"{to_get_count} name")
            self.assertEqual(len(tc.column_items), 1)
            self.assertEqual(len(tc.column_items[0]), to_get_count)
            self.assertLessEqual(set(tc.column_items[0]), name_set)
            tc.reset()
            tc.do_data(f"{to_get_count} name 12")
            self.assertEqual(len(tc.column_items), 1)
            self.assertEqual(len(tc.column_items[0]), to_get_count)
            tc.reset()
            tc.do_data(f"{to_get_count} name 13")
            self.assertEqual(len(tc.column_items), 1)
            self.assertEqual(set(tc.column_items[0]), set(filter(lambda n: 13 <= len(n), name_set)))
            tc.reset()
            tc.do_data(f"{to_get_count} name 16")
            self.assertEqual(len(tc.column_items), 1)
            self.assertEqual(set(tc.column_items[0]), set(filter(lambda n: 16 <= len(n), name_set)))

    def test_list_tables(self):
        """Test that we can list the tables"""
        metadata = MetaData()
        metadata.reflect(self.engine)
        config = {
            "tables": {
                "unique_constraint_test": {
                    "vocabulary_table": True,
                },
                "no_pk_test": {
                    "ignore": True,
                },
            },
        }
        with TestTableCmd(self.dsn, self.schema_name, metadata, config) as tc:
            tc.do_next("unique_constraint_test")
            tc.do_ignore("")
            tc.do_next("person")
            tc.do_vocabulary("")
            tc.reset()
            tc.do_list("")
            person_listed = False
            unique_constraint_test_listed = False
            no_pk_test_listed = False
            for (text, args, kwargs) in tc.messages:
                if args[2] == "person":
                    self.assertFalse(person_listed)
                    person_listed = True
                    self.assertEqual(args[0], " ")
                    self.assertEqual(args[1], "->V")
                elif args[2] == "unique_constraint_test":
                    self.assertFalse(unique_constraint_test_listed)
                    unique_constraint_test_listed = True
                    self.assertEqual(args[0], "V")
                    self.assertEqual(args[1], "->I")
                elif args[2] == "no_pk_test":
                    self.assertFalse(no_pk_test_listed)
                    no_pk_test_listed = True
                    self.assertEqual(args[0], "I")
                    self.assertEqual(args[1], "   ")
                else:
                    self.assertEqual(args[0], " ")
                    self.assertEqual(args[1], "   ")
            self.assertTrue(person_listed)
            self.assertTrue(unique_constraint_test_listed)
            self.assertTrue(no_pk_test_listed)
