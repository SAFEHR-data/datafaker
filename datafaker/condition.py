from abc import ABC
from collections.abc import abstractmethod
from functools import reduce
import parsy
from typing import Self

class Expression(ABC):
    """ Base type for any evaluatable expression """
    @abstractmethod
    def column_names(self) -> list[str]:
        """ List of column names referenced by this expression. """

    @abstractmethod
    def evaluate(self, column_values: dict[str, any]):
        """ Evaluate the expression given the column values specified. """

    @abstractmethod
    def to_sql(self, qualified: str=None) -> str:
        """ Produce a SQL equivalent of this expression. """

    @abstractmethod
    def precedence(self) -> int:
        """
        Level of precedence in the top-level operator.
        From tightest to loosest:
        9: +X -X 
        8: * / %
        7: + - ||
        6: =, !=, <, <>, <=, etc.
        5: IS, LIKE, BETWEEN, IN, EXISTS, IS OF
        4: NOT
        3: AND
        2: OR
        1: Assignment
        """

    def expr_to_sql(self, expr: Self, qualified: str=None):
        """ Output SQL for ``expr``, bracketed if required """
        if expr.precedence() < self.precedence():
            return f"({expr.to_sql(qualified)})"
        return expr.to_sql(qualified)

    def expr_right_to_sql(self, expr: Self, qualified: str=None):
        """ Output SQL for ``expr`` for the right hand side of the expression, bracketed if required """
        if expr.precedence() <= self.precedence():
            return f"({expr.to_sql(qualified)})"
        return expr.to_sql(qualified)


class ColumnReference(Expression):
    def __init__(self, columnname: str):
        self._columnname = columnname

    def column_names(self) -> list[str]:
        return [self._columnname]

    def evaluate(self, column_values: dict[str, any]):
        return column_values[self._columnname]

    def to_sql(self, qualified: str=None) -> str:
        if qualified is None:
            return self.qn_to_sql(self._columnname)
        return self.qn_to_sql(f'{qualified}.{self._columnname}')

    def precedence(self) -> int:
        return 5


class Constant(Expression):
    def __init__(self, value: int | float | str):
        self._value = value

    def column_names(self) -> list[str]:
        return []

    def evaluate(self, _column_values):
        return self._value

    def to_sql(self, _qualified = None):
        if type(self._value) is str:
            return "'" + self._value.replace("'", "''") + "'"
        return str(self._value)

    def precedence(self) -> int:
        return 99


class UnaryExpression(Expression):
    def __init__(self, expr: Expression):
        self._expr = expr

    def column_names(self) -> list[str]:
        return self._expr.column_names()

    def evaluate(self, column_values: dict[str, any]):
        return self.calculate(self._expr.evaluate(column_values))

    @abstractmethod
    def calculate(self, value):
        """ Transform the value """

    def to_sql(self, qualified: str=None) -> str:
        return self.subsql_to_sql(self.expr_to_sql(self._expr, qualified))

    @abstractmethod
    def subsql_to_sql(self, subsql: str) -> str:
        """ Take the SQL from the subexpression and add the unary expression """

    def precedence(self) -> int:
        return 5


class NullTest(UnaryExpression):
    def subsql_to_sql(self, subsql: str) -> str:
        return f'{subsql} IS NULL'

    def calculate(self, value):
        return value is None


class NotNullTest(UnaryExpression):
    def subsql_to_sql(self, subsql: str) -> str:
        return f'{subsql} IS NOT NULL'

    def calculate(self, value):
        return value is not None


class Not(UnaryExpression):
    def subsql_to_sql(self, subsql: str) -> str:
        return f'NOT {subsql}'

    def precedence(self) -> int:
        return 4

    def calculate(self, value):
        return not value


class BinaryExpression(Expression):
    OPERATOR = "@"
    PRECEDENCE = 6

    def __init__(self, exprs: list[Expression]):
        self._exprs = exprs

    def column_names(self) -> list[str]:
        return reduce(
            lambda x, y: x | y,
            map(lambda z: set(z.column_names()), self._exprs),
            set(),
        )

    def evaluate(self, column_values: dict[str, any]):
        return reduce(
            self.calculate,
            map(lambda x: x.evaluate(column_values), self._exprs),
        )

    @abstractmethod
    def calculate(self, lvalue, rvalue):
        """ Transform the values """

    def to_sql(self, qualified: str=None) -> str:
        subsqls = [self.expr_to_sql(self._exprs[0])] + list(map(
            lambda x: self.expr_right_to_sql(x, qualified),
            self._exprs[1:],
        ))
        return f" {self.OPERATOR} ".join(subsqls)

    def precedence(self) -> int:
        return self.PRECEDENCE


class Equals(Expression):
    OPERATOR = "="

    def calculate(self, lvalue, rvalue):
        return lvalue == rvalue


class NotEquals(Expression):
    OPERATOR = "<>"

    def calculate(self, lvalue, rvalue):
        return lvalue != rvalue


class LessThan(Expression):
    OPERATOR = "<"

    def calculate(self, lvalue, rvalue):
        return lvalue < rvalue


class LessThanEqual(Expression):
    OPERATOR = "<="

    def calculate(self, lvalue, rvalue):
        return lvalue <= rvalue


class MoreThan(Expression):
    OPERATOR = ">"

    def calculate(self, lvalue, rvalue):
        return lvalue > rvalue


class MoreThanEqual(Expression):
    OPERATOR = ">="

    def calculate(self, lvalue, rvalue):
        return lvalue >= rvalue


class Plus(Expression):
    OPERATOR = "+"
    PRECEDENCE = 7

    def calculate(self, lvalue, rvalue):
        return lvalue + rvalue


class Minus(Expression):
    OPERATOR = "-"
    PRECEDENCE = 7

    def calculate(self, lvalue, rvalue):
        return lvalue - rvalue


class Concatenate(Expression):
    OPERATOR = "||"
    PRECEDENCE = 7

    def calculate(self, lvalue, rvalue):
        return lvalue + rvalue


class Times(Expression):
    OPERATOR = "*"
    PRECEDENCE = 8

    def calculate(self, lvalue, rvalue):
        return lvalue * rvalue


class Divide(Expression):
    OPERATOR = "/"
    PRECEDENCE = 8

    def calculate(self, lvalue, rvalue):
        return lvalue / rvalue


class Modulo(Expression):
    OPERATOR = "%"
    PRECEDENCE = 8

    def calculate(self, lvalue, rvalue):
        return lvalue % rvalue


class And(Expression):
    OPERATOR = "AND"
    PRECEDENCE = 3

    def calculate(self, lvalue, rvalue):
        return lvalue and rvalue


class Or(Expression):
    OPERATOR = "OR"
    PRECEDENCE = 2

    def calculate(self, lvalue, rvalue):
        return lvalue or rvalue


def integer() -> parsy.Parser:
    """
    Parses an integer, outputting that integer.
    """
    return parsy.regex(r"-?[0-9]+").map(int)

def column_name() -> parsy.Parser:
    """
    Parses a string beginning with a letter or _, or a double-quoted string.
    """
    return parsy.alt(
        parsy.string('"') >> parsy.regex(r'[^"]*') << parsy.string(""),
        parsy.string(r"[a-zA-Z_][0-9a-zA-Z_]*"),
    )
