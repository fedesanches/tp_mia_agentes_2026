"""Tests propios del grupo — Milestone 3, US-04 (análisis de errores).

Blindan la **taxonomía de modos de fallo** de `student_framework/eval/run.py`.
No verifican "que el código corra" —eso ya está validado contra las trazas
reales del dataset— sino que las *decisiones* de diseño no se rompan en
silencio:

  1. El orden de precedencia entre reglas cuando dos señales coinciden.
     Ninguna traza real del dataset tiene dos señales fuertes a la vez, así
     que sólo un caso sintético puede detectar una regresión acá.
  2. Que la repetición de llamadas NUNCA sea criterio de clasificación
     (`apartment-keys` repite el 78 % y resuelve el escenario).
  3. Que `infra_error` se evalúe primero: si se mueve, una corrida con
     credenciales vencidas se cuenta como fallo del agente.
  4. Los casos límite: sin pasos, sin texto de salida, sin diagnóstico.

Más el test de regresión sobre las cuatro trazas reales, versionadas en
`fixtures/` para que el oráculo no dependa de `eval/runs/` (que no se
versiona).

Ejecutar:
    pytest tests/student/test_failure_analysis.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from student_framework.eval.run import (
    FAILURE_THRESHOLDS,
    _render_failure_section,
    build_summary,
    classify_failure,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Salidas exactas de las world-tools (`mia_world/tools.py`) que el
# clasificador reconoce. Se escriben literales para que un cambio de
# redacción en el mundo rompa el test en vez de degradar el diagnóstico.
OUT_OK = "Estás en Estudio. Ves: una alfombra [id: alfombra]."
OUT_ID_INEXISTENTE = "Error: no existe ningún objeto con id 'fantasma'."
OUT_SIN_SALIDA = "Error: no hay salida 'este' desde aquí. Salidas disponibles: sur."
OUT_NOOP = "caja fuerte ya está abierta."


def _step(
    tool_name: str = "look",
    tool_input: str = "{}",
    tool_output: str | None = OUT_OK,
) -> dict[str, Any]:
    """Un `AgentStep` serializado, como el que guarda el arnés."""
    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "error": None,
    }


def _case(**overrides: Any) -> dict[str, Any]:
    """Un registro de caso con los campos que leen el clasificador y el resumen.

    Por defecto: escenario fallido que agotó el presupuesto (`agent_error`
    presente), para que las reglas de frecuencia sean alcanzables. Cada
    test sobreescribe sólo lo que le interesa.
    """
    case: dict[str, Any] = {
        "scenario": "escenario-test",
        "difficulty": "medium",
        "goal_achieved": False,
        "run_error": None,
        "agent_error": "Se alcanzo el limite de iteraciones sin respuesta final.",
        "steps": [],
        "num_tool_calls": 0,
        "latency_seconds": 1.0,
        "input_tokens": 100,
        "output_tokens": 10,
        "rubric": None,
        "failure": None,
    }
    case.update(overrides)
    return case


# --- Una regla por categoría --------------------------------------------------


def test_infra_error_se_evalua_primero():
    """Un fallo del entorno gana aunque la traza tenga otras señales fuertes.

    Es la regla más consecuente: si se mueve de lugar, una corrida con
    credenciales vencidas se contabiliza como fallo del agente.
    """
    caso = _case(
        run_error="Traceback ... ExpiredTokenException",
        steps=[_step("examine", '{"target": "fantasma"}', OUT_ID_INEXISTENTE)] * 5,
    )
    assert classify_failure(caso)["category"] == "infra_error"


def test_terminacion_prematura_cuando_el_bucle_cierra_solo():
    """Sin `agent_error` el bucle no agotó pasos: el agente creyó que terminó."""
    caso = _case(agent_error=None, steps=[_step(), _step("examine")])
    assert classify_failure(caso)["category"] == "terminacion_prematura"


def test_id_alucinado_por_encima_del_umbral():
    """2 de 5 pasos (40 %) sobre ids inexistentes supera el umbral de 20 %."""
    caso = _case(
        steps=[_step("examine", '{"target": "fantasma"}', OUT_ID_INEXISTENTE)] * 2
        + [_step()] * 3
    )
    resultado = classify_failure(caso)
    assert resultado["category"] == "id_alucinado"
    assert resultado["signals"]["missing_id_frac"] >= FAILURE_THRESHOLDS["id_alucinado"]


def test_loop_navegacion_por_encima_del_umbral():
    """2 de 5 pasos (40 %) son `go` fallidos; supera el umbral de 25 %."""
    caso = _case(
        steps=[_step("go", '{"direction": "este"}', OUT_SIN_SALIDA)] * 2
        + [_step()] * 3
    )
    assert classify_failure(caso)["category"] == "loop_navegacion"


def test_loop_improductivo_por_encima_del_umbral():
    """1 de 5 pasos (20 %) sobre un estado ya alcanzado; umbral 10 %."""
    caso = _case(
        steps=[_step("use", '{"item": "llave", "target": "caja"}', OUT_NOOP)]
        + [_step()] * 4
    )
    assert classify_failure(caso)["category"] == "loop_improductivo"


def test_objetivo_incompleto_es_el_fallback():
    """Agotó el presupuesto sin ninguna firma dominante."""
    caso = _case(steps=[_step(), _step("examine"), _step("take")])
    assert classify_failure(caso)["category"] == "objetivo_incompleto"


# --- Las decisiones de diseño (lo que las trazas reales no cubren) ------------


def test_navegacion_gana_sobre_repeticion_y_noop():
    """Con dos señales a la vez manda la más específica, no la más frecuente.

    Reproduce `vault-combination`: 78 % de llamadas repetidas y 65 % de
    movimientos fallidos. Clasificar por frecuencia dominante daría
    `loop_improductivo`, ocultando que el agente estuvo chocando contra
    una pared.
    """
    caso = _case(
        steps=[_step("go", '{"direction": "este"}', OUT_SIN_SALIDA)] * 4
        + [_step("use", '{"item": "llave", "target": "caja"}', OUT_NOOP)] * 2
        + [_step()] * 4
    )
    resultado = classify_failure(caso)
    assert resultado["category"] == "loop_navegacion"
    # Las tres señales están presentes: la precedencia es lo que decide.
    assert resultado["signals"]["nav_error_frac"] >= FAILURE_THRESHOLDS["loop_navegacion"]
    assert resultado["signals"]["noop_frac"] >= FAILURE_THRESHOLDS["loop_improductivo"]
    assert resultado["signals"]["repeated_frac"] >= FAILURE_THRESHOLDS["loop_estancado"]


def test_loop_estancado_solo_actua_como_ultimo_recurso():
    """La repetición clasifica, pero recién después de las causas específicas.

    Es la regla más genérica —la repetición acompaña a casi cualquier
    atasco—, así que sólo describe el fallo cuando ninguna otra lo
    explica. Con la traza toda repetida y sin ninguna otra señal, es la
    que corresponde.
    """
    resultado = classify_failure(_case(steps=[_step()] * 10))
    assert resultado["signals"]["repeated_frac"] >= FAILURE_THRESHOLDS["loop_estancado"]
    assert resultado["category"] == "loop_estancado"


def test_repeticion_por_debajo_del_umbral_cae_en_el_fallback():
    """Repetir poco no alcanza para diagnosticar: queda `objetivo_incompleto`.

    Protege el umbral de la regla 6, calibrado en el hueco 53 %–61 % que
    separa los casos resueltos de los fallidos en 24 corridas.
    """
    caso = _case(steps=[_step(), _step(), _step("examine"), _step("take")])
    resultado = classify_failure(caso)
    assert resultado["signals"]["repeated_frac"] < FAILURE_THRESHOLDS["loop_estancado"]
    assert resultado["category"] == "objetivo_incompleto"


# --- Robustez -----------------------------------------------------------------


def test_caso_sin_pasos_no_rompe():
    """Traza vacía: no debe dividir por cero."""
    resultado = classify_failure(_case(steps=[]))
    assert resultado["category"] == "objetivo_incompleto"
    assert resultado["signals"]["nav_error_frac"] == 0.0


def test_paso_sin_texto_de_salida_no_rompe():
    """Un paso puede no tener `tool_output`; los detectores deben tolerarlo."""
    caso = _case(steps=[_step(tool_output=None), _step("go", "{}", None)])
    assert classify_failure(caso)["category"] == "objetivo_incompleto"


# --- Agregación en el resumen -------------------------------------------------


def test_build_summary_cuenta_solo_los_fallos():
    """El desglose se calcula sobre los casos fallidos, no sobre el total."""
    resuelto = _case(scenario="resuelto", goal_achieved=True, agent_error=None)
    navegacion = _case(
        scenario="navegacion",
        steps=[_step("go", '{"direction": "este"}', OUT_SIN_SALIDA)] * 3,
    )
    alucinado = _case(
        scenario="alucinado",
        steps=[_step("examine", '{"target": "fantasma"}', OUT_ID_INEXISTENTE)] * 3,
    )

    casos = [resuelto, navegacion, alucinado]
    for c in casos:  # replica lo que hace `run_scenario`.
        c["failure"] = classify_failure(c) if not c["goal_achieved"] else None

    resumen = build_summary(casos)
    assert resumen["failures_total"] == 2
    assert resumen["failures_by_category"] == {
        "loop_navegacion": 1,
        "id_alucinado": 1,
    }


def test_build_summary_tolera_casos_sin_diagnostico():
    """Las corridas anteriores a US-04 no traen `failure`: se cuentan sin categoría."""
    viejo = _case(scenario="corrida-vieja")
    del viejo["failure"]

    resumen = build_summary([viejo])
    assert resumen["failures_total"] == 1
    assert resumen["failures_by_category"] == {}


# --- Reporte ------------------------------------------------------------------


def test_reporte_advierte_cuando_hay_infra_error():
    """`infra_error` no es un fallo del agente: la tabla debe decirlo."""
    caso = _case(scenario="cortado", run_error="Traceback ... ExpiredTokenException")
    caso["failure"] = classify_failure(caso)

    texto = "\n".join(_render_failure_section(build_summary([caso]), [caso]))
    assert "infra_error" in texto
    assert "No son fallos del agente" in texto


def test_reporte_sin_fallos_no_imprime_tabla_vacia():
    """Con todo resuelto, una línea es más clara que una tabla con encabezados."""
    caso = _case(scenario="resuelto", goal_achieved=True, agent_error=None)

    texto = "\n".join(_render_failure_section(build_summary([caso]), [caso]))
    assert "Sin fallos en esta corrida" in texto
    assert "| Categoría |" not in texto


# --- Regresión sobre las trazas reales del dataset ---------------------------

CATEGORIAS_ESPERADAS = {
    "extreme-archive": "id_alucinado",
    "library-search": "terminacion_prematura",
    "office-sequence": "loop_improductivo",
    "vault-combination": "loop_navegacion",
}


@pytest.mark.parametrize(
    "escenario,categoria", sorted(CATEGORIAS_ESPERADAS.items())
)
def test_categorias_sobre_trazas_reales(escenario: str, categoria: str):
    """Los 4 fallos reales de la corrida 4/8 deben dar 4 categorías distintas.

    Las trazas están versionadas en `fixtures/` (copiadas de
    `eval/runs/20260827T003846Z/`, que no se versiona) para que este
    oráculo sobreviva a un clon limpio del repositorio.
    """
    caso = json.loads(
        (FIXTURES_DIR / f"{escenario}.json").read_text(encoding="utf-8")
    )
    assert caso["goal_achieved"] is False
    assert classify_failure(caso)["category"] == categoria
