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
    """Construye y configura su agente.

    `config` es opcional. Si se proporciona `config["llm_client"]`, el
    agente debe usarlo (así es como los tests de conformidad inyectan un
    cliente mock). Si no, se construye a partir del entorno.

    TODO (M1): instancien su agente y llamen a `agent.register_tool(...)`
    por cada una de sus herramientas antes de devolverlo.
    """
    
    config = config or {}  # NO CAMBIAR
    llm = config.get("llm_client") or LLMClient.from_env()  # NO CAMBIAR
    kwargs: dict[str, Any] = {"llm_client": llm}  # NO CAMBIAR

    if "max_history_messages" in config:
        kwargs["max_history_messages"] = config["max_history_messages"]

    system_prompt = (
        "Eres un asistente conversacional. "
        "Tienes dos herramientas disponibles: una calculadora y una consulta del clima. "
        "IMPORTANTE: la gran mayoría de las preguntas NO requieren herramientas. "
        "Usa 'calculator' ÚNICAMENTE si el usuario pide calcular una operación matemática con números concretos "
        "(ejemplo: '¿cuánto es 5 + 3?', '¿cuánto es 17 * 4?'). "
        "Usa 'current_temperature' ÚNICAMENTE si el usuario pide el clima o temperatura de una ciudad concreta "
        "(ejemplo: '¿qué temperatura hace en Roma?', '¿cómo está el clima en Tokio?'). "
        "Ante cualquier otra pregunta —saludos, preguntas sobre vos, conversación general— respondé directamente con texto, sin llamar ninguna herramienta. "
        "Ejemplos de preguntas que NUNCA usan herramientas: "
        "'¿Cómo estás?', '¿Quién sos?', '¿Qué podés hacer?', 'Hola', 'Gracias', '¿Cuál es tu nombre?'. "
        "Cuando uses una herramienta, reportá el resultado exacto que devuelve, sin modificarlo."
    )
    agent = MyAgent(system_prompt=system_prompt, **kwargs)

    from student_framework.tools.calculator import calculator, calculator_schema
    from student_framework.tools.weather import current_temperature, current_temperature_schema

    agent.register_tool(calculator, calculator_schema)
    agent.register_tool(current_temperature, current_temperature_schema)

    return agent
