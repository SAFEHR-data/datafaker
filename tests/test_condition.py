""" Tests for the base module. """
from datafaker.condition import integer, column_name, literal_string, expression

from tests.utils import DatafakerTestCase


class ParseTests(DatafakerTestCase):

    def test_simple_parse(self) -> None:
        """Test that simple atoms parse correctly."""
        self.assertEqual(integer.parse("23").evaluate({}), 23)
        self.assertEqual(column_name.parse("one").evaluate({"one": 45, "two": 102}), 45)
        self.assertEqual(column_name.parse('"two"').evaluate({"one": 45, "two": 102}), 102)
        self.assertEqual(literal_string.parse("'literal'").evaluate({"one": 45, "two": 102}), "literal")

    def test_simple_expression(self) -> None:
        """Test that simple expressions parse correctly."""
        self.assertEqual(expression.parse("23").evaluate({}), 23)
        self.assertEqual(expression.parse("one").evaluate({"one": 45, "two": 102}), 45)
        self.assertEqual(expression.parse('"two"').evaluate({"one": 45, "two": 102}), 102)
        self.assertEqual(expression.parse("'literal'").evaluate({"one": 45, "two": 102}), "literal")

    def test_complex_expression(self) -> None:
        """Test that more complex expressions parse correctly."""
        self.assertEqual(expression.parse("2+5").evaluate({}), 7)
        self.assertEqual(expression.parse("2+col").evaluate({"col": 13}), 15)
        self.assertEqual(expression.parse("'prefix '||col").evaluate({"col": "then"}), "prefix then")
        self.assertEqual(expression.parse("2+5*3+4").evaluate({}), 21)
        self.assertEqual(expression.parse("2*5+3*4").evaluate({}), 22)
        self.assertEqual(expression.parse("2*(5+3)*4").evaluate({}), 64)
        self.assertEqual(expression.parse("14+-3").evaluate({}), 11)
