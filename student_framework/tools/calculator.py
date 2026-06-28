"""Calculadora simple para el Milestone 1."""

from __future__ import annotations

from typing import Annotated
from pydantic import Field
from mia_agents.types import ToolSchema


def calculator(
    left_operand:  Annotated[float, Field(description="El primer operando (numero de la izquierda).")],
    right_operand: Annotated[float, Field(description="El segundo operando (numero de la derecha).")],
    operator:      Annotated[str, Field(description="Operador aritmetico a aplicar. Valores validos: '+' (suma), '-' (resta), '*' (multiplicacion), '%' (modulo o resto de la division entera).")],
) -> str:
    """
    Realiza una operacion aritmetica binaria entre dos numeros. 
    Soporta suma (+), resta (-), multiplicacion (*) y modulo (%). 
    Devuelve el resultado como texto. Usar para cualquier calculo matematico basico.
    """
    
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

    return str(result)


calculator_schema = ToolSchema.from_callable(calculator)
