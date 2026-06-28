"""Paquete propio del grupo.

Implementen el agente en `agent.py` y registren sus herramientas a
continuacion, en `build_agent`. Tanto el runner de la CLI como los tests
de conformidad llaman a `build_agent`, por lo que esta es la unica puerta
de entrada publica de su entrega.
"""

from __future__ import annotations

from typing import Any

from mia_agents.llm_client import LLMClient
from mia_agents.protocols import Agent

from .agent import MyAgent


def build_agent(config: dict[str, Any] | None = None) -> Agent:
    """Construye y configura su agente."""
    config = config or {}  # NO CAMBIAR
    llm = config.get("llm_client") or LLMClient.from_env()  # NO CAMBIAR
    kwargs: dict[str, Any] = {"llm_client": llm}  # NO CAMBIAR

    if "max_history_messages" in config:
        kwargs["max_history_messages"] = config["max_history_messages"]

    agent = MyAgent(**kwargs)

    # Registro de lector de archivos
    from student_framework.tools.file_reader import file_reader, file_reader_schema
    agent.register_tool(file_reader, file_reader_schema)
    # Registro calculadora
    from student_framework.tools.calculator import calculator, calculator_schema
    agent.register_tool(calculator, calculator_schema)
    # Registro current_temperature
    from student_framework.tools.weather import current_temperature, current_temperature_schema
    agent.register_tool(current_temperature, current_temperature_schema)

    return agent
