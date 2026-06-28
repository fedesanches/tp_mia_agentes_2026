"""Tests propios del grupo — Milestone 1.

Escenarios deterministas que verifican el bucle del agente y cómo responde
ante casos límite, usando MockLLMClient (sin depender de Ollama/Bedrock).

Cubren:
  1. Encadenamiento de dos herramientas en un mismo run.
  2. Parada por max_iterations (sin bucles infinitos).
  3. Robustez ante herramienta desconocida (alucinada por el LLM).
  4. Robustez ante argumentos JSON inválidos.
  5. El bucle continúa tras una tool y answer proviene del texto del LLM.
  6. El resultado de la herramienta se realimenta al LLM.

Estos tests son complementarios a los de tests/conformance/. 
Aquí validamos nuestro propio contrato de M1.

Ejecutar:
    PYTHONPATH=. pytest tests/student/test_tools_m1.py -v
"""

import json
from mia_agents.testing import MockLLMClient
from mia_agents.types import AgentResult, LLMResponse, ToolCall

from student_framework import build_agent


def test_encadenar_dos_herramientas(tmp_path):
    """
    El LLM llama a file_reader y luego a calculator, y el resultado final
    proviene del texto del LLM.
    Parametros:
        - tmp_path: pytest fixture que provee un directorio temporal para crear archivos. 
    """

    archivo = tmp_path / "numero.txt"
    archivo.write_text("21", encoding="utf-8")

    # Creamos un LLM que primero llama a file_reader, luego a calculator, y finalmente devuelve un texto.
    mock = MockLLMClient([
        LLMResponse(content=None, tool_calls=[
            ToolCall(id="c1", name="file_reader",
                     arguments=json.dumps({"path": str(archivo)}))]),
        LLMResponse(content=None, tool_calls=[
            ToolCall(id="c2", name="calculator",
                     arguments=json.dumps({"left_operand": 21, "right_operand": 2, "operator": "*"}))]),
        LLMResponse(content="El doble es 42."),
    ])
    result = build_agent({"llm_client": mock}).run("leé y multiplicá")

    assert [s.tool_name for s in result.steps] == ["file_reader", "calculator"]
    assert result.steps[0].tool_output == "21"
    assert result.steps[1].tool_output == "42.0"
    assert result.answer == "El doble es 42."
    assert mock.call_count == 3


def test_cortar_por_max_iteraciones():
    """
    El LLM devuelve siempre un tool_call, pero el agente corta tras max_iterations.
    """

    # Creamos un LLM que siempre devuelve un tool_call válido, para simular un bucle infinito.
    tool_call_response  = LLMResponse(content=None, tool_calls=[
        ToolCall(id="c", name="calculator",
                 arguments=json.dumps({"left_operand": 1, "right_operand": 1, "operator": "+"}))])
    mock = MockLLMClient([tool_call_response] * 15)

    result = build_agent({"llm_client": mock}).run("loop infinito")

    assert isinstance(result, AgentResult)
    assert mock.call_count == 10  # default max_iterations


def test_invocar_herramienta_desconocida_sin_romper():
    """
    El LLM llama a una herramienta desconocida, pero el agente no rompe.
    """

    mock = MockLLMClient([
        LLMResponse(content=None, tool_calls=[
            ToolCall(id="c1", name="no_existe", arguments=json.dumps({"x": 1}))]),
        LLMResponse(content="ok"),
    ])
    result = build_agent({"llm_client": mock}).run("usá algo inexistente")

    assert isinstance(result, AgentResult)
    assert any(s.error for s in result.steps)


def test_usar_argumentos_json_invalidos_sin_romper():
    """
    El LLM llama a una herramienta con argumentos JSON inválidos, pero el agente no rompe.
    """

    mock = MockLLMClient([
        LLMResponse(content=None, tool_calls=[
            ToolCall(id="c1", name="calculator", arguments="esto no es json")]),
        LLMResponse(content="ok"),
    ])
    result = build_agent({"llm_client": mock}).run("args rotos")

    assert isinstance(result, AgentResult)
    assert any(s.error for s in result.steps)


def test_respuesta_final_proviene_del_llm():
    """
    El LLM devuelve un tool_call, pero el agente continúa y la respuesta final es del LLM.
    """
    
    mock = MockLLMClient([
        LLMResponse(content=None, tool_calls=[
            ToolCall(id="c1", name="calculator",
                     arguments=json.dumps({"left_operand": 2, "right_operand": 2, "operator": "+"}))]),
        LLMResponse(content="El resultado final es 4."),
    ])
    result = build_agent({"llm_client": mock}).run("sumá 2+2")

    assert result.steps[0].tool_output == "4.0"
    assert result.answer == "El resultado final es 4."
    assert mock.call_count == 2
