""" Tests for the base module. """
from datafaker.condition import integer

from tests.utils import DatafakerTestCase


class ParseTests(DatafakerTestCase):

    def test_simple_parse(self) -> None:
        """Test that simple expressions parse correctly."""
        self.assertEqual(integer.parse("23").evaluate({}), 23)
