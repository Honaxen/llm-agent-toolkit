"""
calculator.py
-------------
Calculator tool for the LLM agent.
Safely evaluates mathematical expressions.
"""

import ast
import operator


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    """Recursively evaluate AST nodes — only safe math operations."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op(_safe_eval(node.operand))
    else:
        raise ValueError(f"Unsupported expression: {type(node)}")


class CalculatorTool:
    """
    Safe mathematical expression evaluator.
    Uses AST parsing instead of eval() to prevent code injection.
    """

    name = "calculator"
    description = "Evaluate mathematical expressions. Input: a math expression like '2 + 2' or '(3 * 4) / 2'."

    def run(self, expression: str) -> str:
        """
        Evaluate a mathematical expression safely.

        Args:
            expression: Math expression string

        Returns:
            Result as string
        """
        try:
            tree = ast.parse(expression.strip(), mode='eval')
            result = _safe_eval(tree.body)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error evaluating '{expression}': {str(e)}"


if __name__ == "__main__":
    tool = CalculatorTool()
    tests = ["2 + 2", "10 / 3", "2 ** 10", "(3 + 4) * 2"]
    for expr in tests:
        print(tool.run(expr))