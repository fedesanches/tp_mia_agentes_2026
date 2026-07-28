"""
Test propios para el grupo — Milestone 2.

Cubre:
    1. Reintentos ante errores transitorios del LLM (TimeoutError, ConnectionError, URLError).
    2. No reintenta ante errores no transitorios (ValueError, TypeError, etc.).
"""

from mia_agents.testing import MockLLMClient
import pytest
from mia_agents.types import LLMResponse
from student_framework import build_agent
from urllib.error import URLError

def test_reintenta_ante_fallo_transitorio_del_llm():
    mock = MockLLMClient([
        TimeoutError("timeout simulado"),
        LLMResponse(content="ok"),
    ])
    agent = build_agent({"llm_client": mock})
    result = agent.run("hola")
    assert result.answer == "ok"
    assert mock.call_count == 2  # 1 fallo + 1 exitoso

def test_no_reintenta_ante_error_no_transitorio():
    mock = MockLLMClient([ValueError("esto no es de red")])
    agent = build_agent({"llm_client": mock})
    try:
        agent.run("hola")
        assert False, "debería haber propagado la excepción"
    except ValueError:
        pass
    assert mock.call_count == 1  # no debió reintentar
