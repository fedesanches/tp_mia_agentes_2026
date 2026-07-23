"""Calculadora simple para el Milestone 1."""

from __future__ import annotations

from typing import Annotated, Any
from pydantic import Field
from mia_agents.types import ToolSchema

# Operadores soportados por la calculadora.
_ALLOWED_OPERATORS = ("+", "-", "*", "%")


def _coerce_operand(name: str, value: Any) -> tuple[float | None, str | None]:
    """Convierte `value` a float o devuelve un mensaje de error accionable.

    Devuelve `(numero, None)` si la conversion fue exitosa, o
    `(None, mensaje)` describiendo que parametro fallo, que valor recibio
    y por que no es valido, para que el LLM pueda corregirlo.
    """
    # `bool` es subclase de `int`; para una calculadora no es un numero valido.
    if isinstance(value, bool):
        return None, (
            f"Error: el parametro '{name}' debe ser numerico, pero recibio el "
            f"booleano {value!r}. Pasa un numero, por ejemplo 3 o 2.5."
        )
    if isinstance(value, (int, float)):
        return float(value), None
    if isinstance(value, str):
        try:
            return float(value.strip()), None
        except ValueError:
            return None, (
                f"Error: el parametro '{name}' debe ser numerico, pero recibio el "
                f"texto {value!r}, que no representa un numero. Pasa un numero, "
                f"por ejemplo 3 o 2.5."
            )
    return None, (
        f"Error: el parametro '{name}' debe ser numerico, pero recibio {value!r} "
        f"(tipo {type(value).__name__}). Pasa un numero, por ejemplo 3 o 2.5."
    )


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
    # Validamos los operandos: si no son numericos devolvemos un mensaje
    # accionable en lugar de lanzar una excepcion.
    left, error = _coerce_operand("left_operand", left_operand)
    if error is not None:
        return error
    right, error = _coerce_operand("right_operand", right_operand)
    if error is not None:
        return error

    if operator not in _ALLOWED_OPERATORS:
        permitidos = ", ".join(repr(op) for op in _ALLOWED_OPERATORS)
        return (
            f"Error: operador {operator!r} no soportado. Operadores permitidos: "
            f"{permitidos}."
        )

    if operator == "+":
        result = left + right
    elif operator == "-":
        result = left - right
    elif operator == "*":
        result = left * right
    else:  # operator == "%"
        if right == 0:
            return (
                "Error: no se puede calcular el modulo por cero: 'right_operand' "
                "es 0. Pasa un divisor distinto de cero."
            )
        result = left % right

    return str(result)


calculator_schema = ToolSchema.from_callable(calculator)
