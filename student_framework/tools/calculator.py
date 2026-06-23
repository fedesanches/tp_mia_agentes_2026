"""Calculadora simple para el Milestone 1."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

Number = int | float


def calculator(
    left_operand: Annotated[Number, Field(description="Primer operando numerico.")],
    right_operand: Annotated[Number, Field(description="Segundo operando numerico.")],
    operator: Annotated[str, Field(description="Operador binario: +, -, * o %.")],
) -> str:
    """Calcula una operacion binaria simple con operadores +, -, * o %."""
    left_operand = float(left_operand)
    right_operand = float(right_operand)

    if operator == "+":
        result = left_operand + right_operand
    elif operator == "-":
        result = left_operand - right_operand
    elif operator == "*":
        result = left_operand * right_operand
    elif operator == "%":
        if right_operand == 0:
            return "Error: no se puede calcular modulo por cero."
        result = left_operand % right_operand
    else:
        return f"Error: operador no soportado: {operator}"

    if result.is_integer():
        return str(int(result))
    return str(result)


calculator_schema = ToolSchema.from_callable(calculator)
