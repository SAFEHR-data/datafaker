from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
import parsy
from typing import Self

class Expression(ABC):
    """ Base type for any evaluatable expression """
    @abstractmethod
    def column_names(self) -> set[str]:
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

    def column_names(self) -> set[str]:
        return {self._columnname}

    def evaluate(self, column_values: dict[str, any]):
        return column_values[self._columnname]

    def to_sql(self, qualified: str=None) -> str:
        if qualified is None:
            return self.qn_to_sql(self._columnname)
        return self.qn_to_sql(f'{qualified}.{self._columnname}')

    def precedence(self) -> int:
        return 5


class ConstantBase(Expression):
    def column_names(self) -> set[str]:
        return set()

    def precedence(self) -> int:
        return 99


class Constant(ConstantBase):
    def __init__(self, value: int | float | str):
        self._value = value

    def evaluate(self, _column_values):
        return self._value

    def to_sql(self, _qualified=None):
        if type(self._value) is str:
            return "'" + self._value.replace("'", "''") + "'"
        return str(self._value)


class UnaryExpression(Expression):
    def __init__(self, expr: Expression):
        self._expr = expr

    def column_names(self) -> set[str]:
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
        return self.PRECEDENCE

    @classmethod
    def re_operator(cls) -> str:
        if cls.OPERATOR[-1].isalpha():
            op = cls.OPERATOR.replace(" ", r"\s+")
            return f"\\s*({op}|{op.lower()})\\b\\s*"
        return f"\\s*{cls.OPERATOR}\\s*"


class PrefixExpression(UnaryExpression):
    """ Base class for prefix unary operators like NOT """


class Not(PrefixExpression):
    PRECEDENCE = 4
    OPERATOR = "NOT"

    def subsql_to_sql(self, subsql: str) -> str:
        return "NOT " + subsql

    def calculate(self, value):
        return not value


class Negate(PrefixExpression):
    OPERATOR = "-"
    PRECEDENCE = 9

    def subsql_to_sql(self, subsql: str) -> str:
        return "-" + subsql

    def calculate(self, value):
        return -value


class PostfixExpression(UnaryExpression):
    """ Base class for postfix expressions like IS NULL """

class IsNull(PostfixExpression):
    OPERATOR = "IS NULL"
    PRECEDENCE = 5

    def subsql_to_sql(self, subsql: str) -> str:
        return subsql + " IS NULL"

    def calculate(self, value):
        return value is None


class IsNotNull(PostfixExpression):
    OPERATOR = "IS NOT NULL"
    PRECEDENCE = 5

    def subsql_to_sql(self, subsql: str) -> str:
        return subsql + " IS NOT NULL"

    def calculate(self, value):
        return value is not None


class BinaryExpression(Expression):
    PRECEDENCE = 6

    def __init__(self, *exprs: list[Expression]):
        self._exprs = exprs

    def column_names(self) -> set[str]:
        return set.union(*(ex.column_names() for ex in self._exprs))

    def evaluate(self, column_values: dict[str, any]):
        return reduce(
            self.calculateMaybeNone,
            map(lambda x: x.evaluate(column_values), self._exprs),
        )

    def calculateMaybeNone(self, lvalue, rvalue):
        """
        Apply the operator.
        In the default implementation either value being None
        produces a result of None.
        """
        if lvalue is None or rvalue is None:
            return None
        return self.calculate(lvalue, rvalue)

    @abstractmethod
    def calculate(self, lvalue, rvalue):
        """
        Apply the operator to lvalue and rvalue.
        Neither lvalue nor rvalue will be None.
        """

    def to_sql(self, qualified: str=None) -> str:
        subsqls = [self.expr_to_sql(self._exprs[0])] + list(map(
            lambda x: self.expr_right_to_sql(x, qualified),
            self._exprs[1:],
        ))
        return f" {self.OPERATOR} ".join(subsqls)

    def precedence(self) -> int:
        return self.PRECEDENCE

    @classmethod
    def re_operator(cls) -> str:
        op = cls.OPERATOR
        if op[-1].isalpha():
            return f"\\s*({op}|{op.lower()})\\b\\s*"
        return f"\\s*{op}\\s*"


class Equals(BinaryExpression):
    OPERATOR = "="

    def calculate(self, lvalue, rvalue):
        return lvalue == rvalue


class NotEquals(BinaryExpression):
    OPERATOR = "<>"

    def calculate(self, lvalue, rvalue):
        return lvalue != rvalue


class LessThan(BinaryExpression):
    OPERATOR = "<"

    def calculate(self, lvalue, rvalue):
        return lvalue < rvalue


class LessThanEqual(BinaryExpression):
    OPERATOR = "<="

    def calculate(self, lvalue, rvalue):
        return lvalue <= rvalue


class MoreThan(BinaryExpression):
    OPERATOR = ">"

    def calculate(self, lvalue, rvalue):
        return lvalue > rvalue


class MoreThanEqual(BinaryExpression):
    OPERATOR = ">="

    def calculate(self, lvalue, rvalue):
        return lvalue >= rvalue


class Plus(BinaryExpression):
    OPERATOR = "+"
    PRECEDENCE = 7

    def calculate(self, lvalue, rvalue):
        return lvalue + rvalue

    @classmethod
    def re_operator(cls) -> str:
        return r"\s*\+\s*"


class Minus(BinaryExpression):
    OPERATOR = "-"
    PRECEDENCE = 7

    def calculate(self, lvalue, rvalue):
        return lvalue - rvalue


class Concatenate(BinaryExpression):
    OPERATOR = "||"
    PRECEDENCE = 7

    def calculate(self, lvalue, rvalue):
        return lvalue + rvalue

    @classmethod
    def re_operator(cls) -> str:
        return r"\s*\|\|\s*"


class Times(BinaryExpression):
    OPERATOR = "*"
    PRECEDENCE = 8

    def calculate(self, lvalue, rvalue):
        return lvalue * rvalue

    @classmethod
    def re_operator(cls) -> str:
        return r"\s*\*\s*"


class Divide(BinaryExpression):
    OPERATOR = "/"
    PRECEDENCE = 8

    def calculate(self, lvalue, rvalue):
        return lvalue / rvalue


class Modulo(BinaryExpression):
    OPERATOR = "%"
    PRECEDENCE = 8

    def calculate(self, lvalue, rvalue):
        return lvalue % rvalue


class And(BinaryExpression):
    OPERATOR = "AND"
    PRECEDENCE = 3

    def calculate(self, lvalue, rvalue):
        return lvalue and rvalue


class Or(BinaryExpression):
    OPERATOR = "OR"
    PRECEDENCE = 2

    def calculate(self, lvalue, rvalue):
        return lvalue or rvalue


""" Parses an integer, outputting that integer. """
integer = parsy.regex(r"-?[0-9]+").map(lambda s: Constant(int(s)))

""" Parses a string beginning with a letter or _, or a double-quoted string. """
column_name = parsy.alt(
    parsy.regex(r'"([^"]*)"', group=1),
    parsy.regex(r"[a-zA-Z_][0-9a-zA-Z_]*"),
).map(ColumnReference)

"""
Parses a literal string, beginning and ending with an apostrophe.
"""
literal_string = parsy.regex(r"'([^']*(''[^']*)*)'", group=1).map(
    lambda s: s.replace("''", "'")
).map(Constant)

""" Parses a unary prefix operator """
prefix_operator = parsy.alt(*(
    parsy.regex(exp.re_operator()).result(exp)
    for exp in (Negate, Not)
))

""" Parses a unary postfix operator """
postfix_operator = parsy.alt(*(
    parsy.regex(exp.re_operator()).result(exp)
    for exp in (IsNull, IsNotNull)
))

BINARY_EXPRESSIONS = [
    Equals,
    NotEquals,
    LessThan,
    LessThanEqual,
    MoreThan,
    MoreThanEqual,
    Plus,
    Minus,
    Concatenate,
    Times,
    Divide,
    Modulo,
    And,
    Or,
]


""" Parses a binary operator """
binary_operator = parsy.alt(*(
    parsy.regex(exp.re_operator()).result(exp)
    for exp in BINARY_EXPRESSIONS
))

bracketed_expression = parsy.forward_declaration()

term = parsy.alt(
    bracketed_expression,
    integer,
    literal_string,
    column_name,
)

unaries = parsy.seq(
    prefix_operator.many(),
    term,
    postfix_operator.many(),
)


class ExpressionBuilder:
    @dataclass
    class ItemBase:
        """ Base class for items in the stack of expressions yet to apply """

    @dataclass
    class ItemPrefix(ItemBase):
        """ Represents a prefix like NOT """
        pref: type[PrefixExpression]

        def precedence(self):
            return self.pref.PRECEDENCE

        def combine(self, t: Expression):
            return self.pref(t)

    @dataclass
    class ItemBinary(ItemBase):
        """ Represents a binary operator and the expression on its left """
        exp: Expression
        op: type[BinaryExpression]

        def precedence(self) -> int:
            return self.op.PRECEDENCE

        def combine(self, t: Expression):
            return self.op(self.exp, t)

    def __init__(self):
        self._stack: list[ItemBase] = []

    def add_prefix(self, p: type[PrefixExpression]) -> None:
        """ Stack a prefix operator """
        self._stack.append(self.ItemPrefix(p))

    def add_prefixes(self, ps: list[type[PrefixExpression]]) -> None:
        """ Stack some prefix operators """
        for p in ps:
            self.add_prefix(p)

    def combine(self, t: Expression, precedence: int) -> Expression:
        """
        Combine all items on the end of the stack
        whose operators have precedence greater than or equal to `precedence`
        with `t`.
        """
        while self._stack and precedence <= self._stack[-1].precedence():
            s = self._stack.pop()
            t = s.combine(t)
        return t

    def combine_postfix(self, t: Expression, postfix: type[PostfixExpression]) -> Expression:
        """
        Combine `t` with the postfix operator `post`
        having combined `t` with any tighter expressions on its left
        """
        t = self.combine(t, postfix.PRECEDENCE)
        return postfix(t)

    def combine_postfixes(self, t: Expression, postfixes: list[type[PostfixExpression]]) -> Expression:
        """
        Combine `t` with the postfix operators in `postfixes`
        having combined `t` with any tighter expressions on its left
        """
        for p in postfixes:
            t = self.combine_postfix(t, p)
        return t

    def add_binary(self, t: Expression, bop: type[BinaryExpression]) -> None:
        """
        Add this expression and the binary operator on its right to the stack.
        """
        self._stack.append(self.ItemBinary(t, bop))


@parsy.generate
def expression():
    """
    Parse an expression, which is terms linked by binary operators.
    """
    (prefixes, t, postfixes) = yield unaries
    rest = yield parsy.seq(binary_operator, unaries).many()

    builder = ExpressionBuilder()

    for (bop, next_term) in rest:
        builder.add_prefixes(prefixes)
        t = builder.combine_postfixes(t, postfixes)
        t = builder.combine(t, bop.PRECEDENCE)
        builder.add_binary(t, bop)
        (prefixes, t, postfixes) = next_term

    builder.add_prefixes(prefixes)
    t = builder.combine_postfixes(t, postfixes)
    t = builder.combine(t, 0)

    return t


""" Parse an expression in brackets """
bracketed_expression.become(parsy.regex(r"\( *") >> expression << parsy.regex(r" *\)"))

def parse_expression(expression_string: str) -> Expression:
    """ Parse a SQL-like expression, returning an Expression object """
    return expression.parse(expression_string)
