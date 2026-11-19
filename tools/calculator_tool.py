"""A small arithmetic tool that never evaluates arbitrary Python code."""

import ast
import operator
from typing import TypeAlias

from langchain_core.tools import tool

Number: TypeAlias = int | float

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate_expression(expression: str) -> Number:
    """Parse and calculate a bounded arithmetic expression."""
    if not expression.strip() or len(expression) > 256:
        raise ValueError("Provide a non-empty expression of at most 256 characters.")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Expression is not valid arithmetic.") from exc

    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("Expression is too complex.")
    result = _evaluate(tree.body, depth=0)
    if isinstance(result, float) and not (float("-inf") < result < float("inf")):
        raise ValueError("Expression result must be finite.")
    return result


def _evaluate(node: ast.AST, depth: int) -> Number:
    if depth > 20:
        raise ValueError("Expression is too deeply nested.")

    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand, depth + 1))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left, depth + 1)
        right = _evaluate(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent magnitude cannot exceed 100.")
        try:
            result = _BINARY_OPERATORS[type(node.op)](left, right)
        except (ArithmeticError, OverflowError) as exc:
            raise ValueError(f"Arithmetic operation failed: {exc}") from exc
        if type(result) not in (int, float):
            raise ValueError("Complex or non-numeric results are not supported.")
        if isinstance(result, int) and result.bit_length() > 4096:
            raise ValueError("Integer result is too large.")
        return result

    raise ValueError("Only numeric constants and basic arithmetic are supported.")


@tool
def calculator(expression: str) -> str:
    """Calculate an arithmetic expression using +, -, *, /, //, %, and **."""
    return str(calculate_expression(expression))
