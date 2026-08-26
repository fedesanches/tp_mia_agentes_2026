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
        "Entre tus herramientas puede haber una calculadora, una consulta del clima "
        "y un lector de archivos de texto, además de otras que se te indiquen según el contexto. "
        "IMPORTANTE: la gran mayoría de las preguntas NO requieren herramientas. "
        "Usa 'calculator' ÚNICAMENTE si el usuario pide calcular una operación matemática con números concretos "
        "(ejemplo: '¿cuánto es 5 + 3?', '¿cuánto es 17 * 4?'). "
        "Usa 'current_temperature' ÚNICAMENTE si el usuario pide el clima o temperatura de una ciudad concreta "
        "(ejemplo: '¿qué temperatura hace en Roma?', '¿cómo está el clima en Tokio?'). "
        "Usa 'file_reader' ÚNICAMENTE si el usuario pide leer el contenido de un archivo indicando su ruta "
        "(ejemplo: 'leé el archivo notas.txt', '¿qué dice data/config.txt?'). "
        "Ante cualquier otra pregunta —saludos, preguntas sobre vos, conversación general— respondé directamente con texto, "
        "sin llamar ninguna herramienta. "
        "Excepción: si tenés disponibles herramientas para explorar un entorno "
        "(por ejemplo para mirar un lugar, examinar objetos, tomarlos, usarlos o "
        "moverte de un lugar a otro) y el usuario te pide resolver una tarea de ese "
        "tipo, seguí siempre este orden: primero observá tu entorno con la "
        "herramienta correspondiente para conocer los objetos realmente presentes "
        "(no asumas ni inventes nombres); luego actuá usando exactamente los "
        "identificadores que esa observación te reportó. Si un objeto parece oculto "
        "dentro de otro, examiná el contenedor antes de intentar tomarlo. "
        "Reglas para resolver estos entornos: "
        "(1) Al entrar a una sala nueva, mirá primero; usá SOLO ids que mirar o "
        "examinar te hayan reportado en la sala actual. Si una acción devuelve 'no "
        "existe ningún objeto con id X' o 'no ves ningún X aquí', ese id es inválido "
        "o estás en la sala equivocada: no vuelvas a intentarlo igual, volvé a mirar "
        "o movete a la sala correcta. "
        "(2) Cadenas de llaves: si una llave no encaja en tu objetivo final, "
        "seguramente abre OTRA cerradura (del mismo color, tamaño o etiqueta). "
        "Después de abrir un contenedor, volvé a examinarlo para revelar su "
        "contenido y tomá lo que haya adentro: suele ser la llave o pieza del "
        "siguiente paso. "
        "(3) Cerraduras de varias piezas: si una cerradura, panel o puerta pide "
        "varias piezas o tiene varias ranuras, juntá TODAS las piezas (pueden estar "
        "en salas distintas) y usá cada una sobre ese mismo objetivo; recién abre "
        "cuando están todas colocadas. "
        "(4) Orden: leé la tarea con cuidado. Si pide hacer algo ANTES que otra cosa "
        "(por ejemplo llevarte un objeto antes de abrir una puerta que se sella), "
        "respetá ese orden: algunas acciones son irreversibles. "
        "(5) No repitas una acción que ya hiciste con éxito ni una que ya falló con "
        "los mismos argumentos; si te encontrás repitiendo, replanteá qué te falta "
        "para el objetivo. "
        "(6) Cuando el objetivo esté cumplido, respondé con un texto final breve y "
        "dejá de llamar herramientas; no sigas explorando en círculos. "
        "Nunca describas en texto que vas a llamar una herramienta ni escribas un "
        "JSON simulando la llamada: invocá la herramienta directamente mediante el "
        "mecanismo de function-calling disponible, sin narrar tu plan antes. "
        "Ejemplos de preguntas que NUNCA usan herramientas: "
        "'¿Cómo estás?', '¿Quién sos?', '¿Qué podés hacer?', 'Hola', 'Gracias', '¿Cuál es tu nombre?'. "
        "Cuando uses una herramienta, reportá el resultado exacto que devuelve, sin modificarlo."
    )
        
    kwargs["system_prompt"] = system_prompt

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
