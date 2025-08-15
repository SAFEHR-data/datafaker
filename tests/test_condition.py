""" Tests for the base module. """
from datafaker.condition import integer, column_name, literal_string, parse_expression

from tests.utils import DatafakerTestCase


class ParseTests(DatafakerTestCase):

    def test_simple_parse(self) -> None:
        """Test that simple atoms parse correctly."""
        self.assertEqual(integer.parse("23").evaluate({}), 23)
        self.assertEqual(column_name.parse("one").evaluate({"one": 45, "two": 102}), 45)
        self.assertEqual(column_name.parse('"two"').evaluate({"one": 45, "two": 102}), 102)
        self.assertEqual(literal_string.parse("'literal'").evaluate({"one": 45, "two": 102}), "literal")

    def test_simple_expression_parsed(self) -> None:
        """Test that simple expressions parse correctly."""
        self.assertEqual(parse_expression("23").evaluate({}), 23)
        self.assertEqual(parse_expression("one").evaluate({"one": 45, "two": 102}), 45)
        self.assertEqual(parse_expression('"two"').evaluate({"one": 45, "two": 102}), 102)
        self.assertEqual(parse_expression("'literal'").evaluate({"one": 45, "two": 102}), "literal")

    def test_complex_expression_parsed(self) -> None:
        """Test that more complex expressions parse correctly."""
        self.assertEqual(parse_expression("2+5").evaluate({}), 7)
        self.assertEqual(parse_expression("2+col").evaluate({"col": 13}), 15)
        self.assertEqual(parse_expression("'prefix '||col").evaluate({"col": "then"}), "prefix then")
        self.assertEqual(parse_expression("2+5*3+4").evaluate({}), 21)
        self.assertEqual(parse_expression("2*5+3*4").evaluate({}), 22)
        self.assertEqual(parse_expression("2*(5+3)*4").evaluate({}), 64)
        self.assertEqual(parse_expression("14+-3").evaluate({}), 11)
        exp = parse_expression("age < 18 AND occupation IS NOT NULL")
        self.assertEqual(exp.column_names(), {"age", "occupation"})
        self.assertFalse(exp.evaluate({"age": 15, "occupation": None}))
        self.assertTrue(exp.evaluate({"age": 15, "occupation": "chimneysweep"}))
        self.assertFalse(exp.evaluate({"age": 45, "occupation": None}))
        self.assertFalse(exp.evaluate({"age": 45, "occupation": "banker"}))
        self.assertFalse(exp.evaluate({"age": 45, "occupation": "waiter"}))

    def test_spaces_parsed_but_not_output(self) -> None:
        """ Test that a condition with lots of spaces is output in a standard form """
        self.assertEqual(parse_expression("2    +5  ").to_sql(), "2 + 5")
        self.assertEqual(parse_expression("   2+  (5 *  4   )").to_sql(), "2 + 5 * 4")
        self.assertEqual(parse_expression("   one   and  two  is   not    null  ").to_sql(), "one AND two IS NOT NULL")
        self.assertEqual(parse_expression('"_one_1_" and "two**three"').to_sql(), '_one_1_ AND "two**three"')
        self.assertEqual(parse_expression('"_one_1_" and "two**three"').to_sql("qu"), 'qu._one_1_ AND qu."two**three"')
