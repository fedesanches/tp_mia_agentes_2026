"""Arnés de evaluación reproducible para el mundo simulado de M3.

Ejecuta el agente del grupo sobre **todo** el dataset de escenarios sin
pasos manuales, capturando por caso: el mensaje de entrada, la respuesta
final, cada llamada a herramienta (input/output/error) y cualquier fallo.
Deja un registro por caso en disco, un resumen agregado con métricas y un
reporte legible.

Además de medir *cuánto* resuelve el agente, califica *cómo* lo resuelve
(rúbrica de calidad de proceso) y *por qué* falla cuando falla (taxonomía
de modos de fallo). Ambas son deterministas sobre la traza y no consumen
tokens, así que dos corridas sobre los mismos datos dan lo mismo.

Uso típico (sin argumentos: corre los 8 escenarios):

    python student_framework/eval/run.py

Opciones útiles:

    python student_framework/eval/run.py --scenario easy      # un escenario
    python student_framework/eval/run.py --max-iterations 40  # tope de pasos
    python student_framework/eval/run.py --repeat 3           # promediar N corridas
    python student_framework/eval/run.py --out-dir <dir>      # dónde guardar
    python student_framework/eval/run.py --reclassify <dir>   # rediagnosticar
                                                              # una corrida ya
                                                              # ejecutada, sin LLM

Salidas (bajo `student_framework/eval/runs/<timestamp>/`):
  - `cases/<scenario_id>.json`  — un registro completo por escenario.
  - `summary.json`              — métricas agregadas legibles por máquina.
  - `report.md`                 — resumen legible por humanos.

Requisitos: el módulo del agente (por defecto `student_framework`) debe
exportar `build_agent`, y debe haber un proveedor LLM configurado (Bedrock
con `BEDROCK_MODEL_ID`, u Ollama con `OLLAMA_HOST`). Ver el README.

Organización del módulo, de lo puro a lo que toca el mundo:

    1. Configuración y datos del dominio
    2. Lectura de trazas — predicados compartidos
    3. Calidad de proceso — la rúbrica (0–8)
    4. Modos de fallo — la taxonomía
    5. Ejecución de un escenario
    6. Agregación de métricas
    7. Reporte legible
    8. Orquestación y CLI
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# `run.py` vive en `student_framework/eval/`; para poder importar
# `student_framework` y `mia_world` al correrlo como script (`python
# student_framework/eval/run.py`) hay que anclar la raíz del repo en sys.path.
EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib  # noqa: E402  (tras ajustar sys.path)

from mia_world.goals import check_goal  # noqa: E402
from mia_world.scenarios import list_scenarios, load_scenario  # noqa: E402
from mia_world.state import Scenario  # noqa: E402
from mia_world.tools import make_world_tools  # noqa: E402
from mia_agents._env import load_env_files  # noqa: E402


# =============================================================================
# 1. Configuración y datos del dominio
# =============================================================================

DEFAULT_SCENARIOS_DIR = REPO_ROOT / "scenarios"
DEFAULT_OUT_DIR = EVAL_DIR / "runs"
DEFAULT_MODULE = "student_framework"

# El bucle del agente corta en `max_iterations`. El peor caso del dataset
# ronda las 21 llamadas óptimas; damos margen para exploración subóptima.
DEFAULT_MAX_ITERATIONS = 40

# Llamadas óptimas por escenario (de la tabla de ENUNCIADO_M3.md). Sirven
# como línea base para la métrica de eficiencia (llamadas vs. óptimo). No
# viven en los JSON de escenario (que son FIJOS), así que se mapean aquí.
OPTIMAL_CALLS: dict[str, int] = {
    "study-with-key": 3,
    "color-locks": 11,
    "apartment-keys": 7,
    "library-search": 7,
    "office-sequence": 13,
    "extreme-archive": 4,
    "vault-combination": 21,
    "backtracking-vault": 18,
}

# =============================================================================
# 2. Lectura de trazas — predicados compartidos
# =============================================================================
#
# Las dos métricas de este módulo (la rúbrica y la taxonomía) leen la misma
# materia prima: la lista de `steps` que el agente deja en `AgentResult`. Estos
# predicados son el vocabulario común para interrogarla — cada uno mira UN paso
# y responde una pregunta.
#
# Dependen de las cadenas exactas que emiten las world-tools de `mia_world`. Un
# cambio de redacción allá degradaría el diagnóstico en silencio; por eso los
# tests fijan esas cadenas como constantes.


def _call_identity(step: dict[str, Any]) -> tuple[str | None, str]:
    """Identidad normalizada de una llamada, para detectar repeticiones.

    Normaliza los argumentos JSON (claves ordenadas) para que dos llamadas
    equivalentes escritas distinto cuenten como la misma.
    """
    name = step.get("tool_name")
    raw = step.get("tool_input")
    try:
        args = json.loads(raw) if raw else {}
        norm = json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        norm = raw or ""
    return (name, norm)


def _is_error_output(step: dict[str, Any]) -> bool:
    """True si el paso falló, por excepción del bucle o por `Error: ...`.

    Las world-tools no lanzan excepciones: devuelven el error como texto.
    Por eso no alcanza con mirar `step["error"]`, que sólo se llena cuando
    la herramienta revienta de verdad.
    """
    if step.get("error"):
        return True
    out = (step.get("tool_output") or "").strip().lower()
    return out.startswith("error:")


def _is_missing_id_output(step: dict[str, Any]) -> bool:
    """True si el agente nombró un objeto que no existe o no puede ver.

    Deliberadamente NO incluye "no llevas ningún X" (usar algo que no está
    en el inventario): eso aparece también en corridas exitosas, así que no
    distingue un fallo de una exploración normal.
    """
    out = (step.get("tool_output") or "").lower()
    return "no existe ning" in out or "no ves ning" in out


def _is_nav_error(step: dict[str, Any]) -> bool:
    """True si un `go` falló: el agente intentó moverse y no se movió.

    Cubre las cuatro formas de fallo de `go` (sin salidas, dirección
    inválida, paso bloqueado, sala desconocida). Todas devuelven un
    `Error: ...` y todas significan lo mismo para el diagnóstico, así que
    se detectan por el prefijo y no por la frase concreta.
    """
    if step.get("tool_name") != "go":
        return False
    return (step.get("tool_output") or "").strip().lower().startswith("error:")


def _is_noop_output(step: dict[str, Any]) -> bool:
    """True si la acción no cambió nada porque ya estaba hecha.

    "ya está abierta" (`mia_world/tools.py`) es hoy el único mensaje de
    no-op del mundo. El detector es exacto y completo, pero frágil: si se
    agregan otros mensajes de este tipo, hay que sumarlos acá.
    """
    return "ya está abierta" in (step.get("tool_output") or "").lower()


# =============================================================================
# 3. Calidad de proceso — la rúbrica (0–8)
# =============================================================================
#
# Métrica cualitativa: puntúa CÓMO resolvió el agente, no si resolvió. Cuatro
# dimensiones de 0–2 sobre la traza de acciones (nunca sobre el texto libre del
# agente, que puede afirmar cualquier cosa).
#
# Se eligió una rúbrica programática antes que un juez LLM por tres razones:
# es reproducible (misma traza → mismo puntaje), no cuesta tokens ni latencia,
# y cada punto se deriva de una regla inspeccionable. A cambio mide sólo
# comportamiento observable, no la calidad del razonamiento interno.

RUBRIC_MAX = 8
_EXPLORE_TOOLS = {"look", "examine"}
_ACTION_TOOLS = {"use", "take"}


def _score_exploration_before_action(steps: list[dict[str, Any]]) -> int:
    """R1: ¿exploró (`look`/`examine`) antes de la primera acción?"""
    action_idx = next(
        (i for i, s in enumerate(steps) if s.get("tool_name") in _ACTION_TOOLS),
        None,
    )
    explored_any = any(s.get("tool_name") in _EXPLORE_TOOLS for s in steps)
    if action_idx is None:
        return 1 if explored_any else 0
    if any(steps[i].get("tool_name") in _EXPLORE_TOOLS for i in range(action_idx)):
        return 2
    return 1 if explored_any else 0


def _score_no_redundant_actions(steps: list[dict[str, Any]]) -> int:
    """R2: penaliza repetir la misma llamada `(tool, args)` idéntica."""
    if not steps:
        return 0
    ids = [_call_identity(s) for s in steps]
    dup = len(ids) - len(set(ids))
    if dup == 0:
        return 2
    return 1 if dup / len(ids) <= 0.25 else 0


def _score_no_hallucinated_ids(steps: list[dict[str, Any]]) -> int:
    """R3: penaliza invocar objetos inexistentes / no visibles."""
    if not steps:
        return 0
    bad = sum(1 for s in steps if _is_missing_id_output(s))
    if bad == 0:
        return 2
    return 1 if bad / len(steps) <= 0.25 else 0


def _score_error_recovery(steps: list[dict[str, Any]]) -> int:
    """R4: tras un paso con error, ¿cambió de acción en vez de insistir?"""
    if not steps:
        return 0
    error_idxs = [i for i, s in enumerate(steps) if _is_error_output(s)]
    if not error_idxs:
        return 2  # nada que recuperar: no se penaliza.
    recovered = 0
    for i in error_idxs:
        nxt = _call_identity(steps[i + 1]) if i + 1 < len(steps) else None
        if nxt is not None and nxt != _call_identity(steps[i]):
            recovered += 1
    frac = recovered / len(error_idxs)
    if frac >= 0.8:
        return 2
    return 1 if frac >= 0.4 else 0


def score_rubric(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Puntaje de calidad de proceso (0–8) sobre la traza de `steps`."""
    r1 = _score_exploration_before_action(steps)
    r2 = _score_no_redundant_actions(steps)
    r3 = _score_no_hallucinated_ids(steps)
    r4 = _score_error_recovery(steps)
    total = r1 + r2 + r3 + r4
    return {
        "exploration_before_action": r1,
        "no_redundant_actions": r2,
        "no_hallucinated_ids": r3,
        "error_recovery": r4,
        "total": total,
        "max": RUBRIC_MAX,
        "normalized": round(total / RUBRIC_MAX, 3),
    }


# =============================================================================
# 4. Modos de fallo — la taxonomía
# =============================================================================
#
# Clasifica POR QUÉ falló un caso, complementando la métrica binaria de éxito:
# la tasa de éxito dice cuánto falla el agente, esto dice de qué. Igual que la
# rúbrica, es determinista sobre la traza y no consume tokens.
#
# Precedencia ORDENADA, de específico a genérico: gana la primera que dispara.
#
#   1. infra_error            run_error is not None (fallo del entorno, no del agente)
#   2. terminacion_prematura  goal falso ∧ agent_error is None ∧ run_error is None
#   3. id_alucinado           ≥20% de pasos con id inexistente / no visible
#   4. loop_navegacion        ≥25% de pasos `go` que devolvieron error
#   5. loop_improductivo      ≥10% de pasos con output de no-op ("ya está abierta")
#   6. loop_estancado         ≥55% de llamadas repetidas (tras agotar el presupuesto)
#   7. objetivo_incompleto    (fallback)
#
# CALIBRACIÓN (reglas 3-5), sobre runs/20260827T003846Z — 8 casos, 4 fallos.
# Cada umbral cae en un hueco vacío de esos datos:
#   id_alucinado      archive 50%  vs  máx. resto  5%   (margen 10x)
#   loop_navegacion   vault   65%  vs  máx. resto 10%   (margen 6.5x)
#   loop_improductivo office  15%  vs  máx. resto  0%   (margen ∞)
# "no llevas ningún X" queda EXCLUIDO de id_alucinado: dispara 20% en
# color-locks y 17% en study-with-key, ambos exitosos.
#
# RECALIBRACIÓN (regla 6), sobre runs/20260827T221119Z — 24 corridas, 9 fallos.
# Con sólo 4 fallos la repetición de llamadas parecía no discriminar
# (`apartment-keys` repetía 78% y resolvía), así que se medía como evidencia
# pero no clasificaba. Con 24 corridas esa decisión no se sostuvo: 6 de 9 fallos
# caían en el fallback y TODOS repetían ≥61%, mientras los resueltos median 29%
# (máx. 53%, con un único caso en 65%). De ahí la regla 6, con el umbral en el
# hueco 53%–61%. Dos salvedades:
#   · Margen estrecho frente a las reglas 3-5; revisar con más corridas.
#   · La regla 6 sólo se alcanza tras descartar 1-5, o sea con el presupuesto
#     agotado. Describe "se quedó sin pasos dando vueltas", no "repitió mucho":
#     repetir NO predice fallar (hay un caso resuelto al 65%), pero sí describe
#     CÓMO se agotó el presupuesto.

FAILURE_THRESHOLDS = {
    "id_alucinado": 0.20,
    "loop_navegacion": 0.25,
    "loop_improductivo": 0.10,
    "loop_estancado": 0.55,
}


def _failure_signals(case: dict[str, Any]) -> dict[str, Any]:
    """Mide las señales de la traza: fracciones de cada tipo de problema.

    Recorre `steps` una sola vez. Las fracciones se calculan sobre el
    total de pasos (no sobre los pasos de cada verbo): separa mejor los
    casos y evita dividir por cero en los escenarios de una sola sala,
    que no registran `go`. Con `steps` vacío todas las fracciones son 0.

    Se devuelven conteos además de fracciones para que la evidencia del
    diagnóstico pueda citar "26/40" y no sólo "65%": el absoluto da la
    escala, que en este dataset varía entre 6 y 40 pasos.
    """
    steps = case.get("steps") or []
    total = len(steps)

    def _frac(count: int) -> float:
        return round(count / total, 3) if total else 0.0

    missing_id = sum(1 for s in steps if _is_missing_id_output(s))
    nav_error = sum(1 for s in steps if _is_nav_error(s))
    noop = sum(1 for s in steps if _is_noop_output(s))

    identities = [_call_identity(s) for s in steps]
    repeated = len(identities) - len(set(identities))

    return {
        "num_steps": total,
        "missing_id_count": missing_id,
        "missing_id_frac": _frac(missing_id),
        "nav_error_count": nav_error,
        "nav_error_frac": _frac(nav_error),
        "noop_count": noop,
        "noop_frac": _frac(noop),
        # Evidencia, NO criterio de clasificación: `apartment-keys` repite
        # el 78 % de sus llamadas y resuelve el escenario.
        "repeated_count": repeated,
        "repeated_frac": _frac(repeated),
        "hit_iteration_limit": bool(case.get("agent_error")),
    }




def classify_failure(case: dict[str, Any]) -> dict[str, Any]:
    """Diagnostica POR QUÉ falló un caso.

    Las reglas, los umbrales y su calibración están en el bloque
    "Modos de fallo — la taxonomía", más arriba en este módulo.
    Se evalúan EN ORDEN, de específico a genérico, y devuelve la primera
    que dispara.

    Nunca devuelve `None`, pero la categoría sólo tiene sentido sobre un
    caso fallido: quién la llama decide cuándo corresponde (ver
    `run_scenario`). Para medir señales sobre un caso resuelto, usá
    `_failure_signals` directamente.
    """
    signals = _failure_signals(case)
    total = signals["num_steps"]

    def _diagnosis(category: str, evidence: str) -> dict[str, Any]:
        return {"category": category, "evidence": evidence, "signals": signals}

    # 1. Fallo del entorno, no del agente. Va primero porque con
    #    `run_error` el agente no llega a correr: `steps` queda vacío y
    #    `agent_error` en None, así que sin este guard el caso caería en
    #    `objetivo_incompleto` y se contaría como fallo del agente.
    if case.get("run_error"):
        return _diagnosis(
            "infra_error",
            "la corrida falló antes de poder evaluar al agente",
        )

    # 2. El bucle terminó por su cuenta: el agente creyó que había
    #    terminado. `AgentResult.error` se setea ÚNICAMENTE al agotar
    #    `max_iterations` (ver `MyAgent.run`), así que su ausencia en un
    #    caso fallido equivale a un cierre voluntario. Si alguien agrega
    #    otro `error=` en el bucle del agente, esta regla deja de valer.
    if not case.get("agent_error"):
        return _diagnosis(
            "terminacion_prematura",
            f"el bucle cerró solo tras {total} llamada(s), sin agotar el presupuesto",
        )

    # 3-6. Señales de frecuencia, de específico a genérico.
    if signals["missing_id_frac"] >= FAILURE_THRESHOLDS["id_alucinado"]:
        return _diagnosis(
            "id_alucinado",
            f"{signals['missing_id_count']}/{total} pasos sobre ids "
            f"inexistentes o no visibles ({signals['missing_id_frac']:.0%})",
        )

    if signals["nav_error_frac"] >= FAILURE_THRESHOLDS["loop_navegacion"]:
        return _diagnosis(
            "loop_navegacion",
            f"{signals['nav_error_count']}/{total} llamadas `go` "
            f"devolvieron error ({signals['nav_error_frac']:.0%})",
        )

    if signals["noop_frac"] >= FAILURE_THRESHOLDS["loop_improductivo"]:
        return _diagnosis(
            "loop_improductivo",
            f"{signals['noop_count']}/{total} acciones sobre un estado ya "
            f"alcanzado ({signals['noop_frac']:.0%})",
        )

    # 6. Última señal, la más genérica: agotó el presupuesto dando vueltas
    #    sobre sus propios pasos. Va al final justamente por genérica: la
    #    repetición acompaña a casi cualquier atasco, así que sólo describe
    #    el fallo cuando ninguna causa más específica lo explica.
    if signals["repeated_frac"] >= FAILURE_THRESHOLDS["loop_estancado"]:
        return _diagnosis(
            "loop_estancado",
            f"{signals['repeated_count']}/{total} llamadas repetidas "
            f"({signals['repeated_frac']:.0%}) hasta agotar el presupuesto",
        )

    # 7. Fallback: agotó el presupuesto sin una firma dominante.
    return _diagnosis(
        "objetivo_incompleto",
        f"agotó el presupuesto en {total} llamadas sin una causa dominante",
    )


# =============================================================================
# 5. Ejecución de un escenario
# =============================================================================
#
# A partir de acá el código toca el mundo: importa el módulo del agente,
# instancia un mundo mutable y llama al LLM. Todo lo anterior son funciones
# puras sobre datos ya capturados.


def _resolve_scenarios(spec: str | None, scenarios_dir: Path) -> list[Scenario]:
    """Devuelve la lista de escenarios a evaluar.

    Sin `spec`, evalúa todo el dataset (ordenado por nombre de fichero).
    Con `spec`, filtra por path, id o dificultad (mismo criterio que la
    CLI de `mia_world`).
    """
    if spec is None:
        return list_scenarios(scenarios_dir)

    path = Path(spec)
    if path.is_file():
        return [load_scenario(path)]

    available = list_scenarios(scenarios_dir)
    by_id = {sc.id: sc for sc in available}
    if spec in by_id:
        return [by_id[spec]]

    by_diff = [sc for sc in available if sc.difficulty == spec]
    if by_diff:
        return by_diff

    options = ", ".join(sorted(sc.id for sc in available)) or "(ninguno)"
    raise SystemExit(
        f"No se encontró el escenario {spec!r}. Disponibles: {options}."
    )


def _build_agent(module_name: str, max_iterations: int) -> Any:
    """Construye el agente vía `build_agent` y ajusta el tope de pasos.

    Reusa el único punto de entrada público del framework (igual que la
    CLI de `mia_world` y los tests de conformidad). `build_agent` ignora
    `max_iterations`, así que lo sobreescribimos sobre la instancia para
    permitir escenarios de horizonte largo y los experimentos que varían
    ese presupuesto.
    """
    module = importlib.import_module(module_name)
    if not hasattr(module, "build_agent"):
        raise SystemExit(f"El módulo {module_name!r} no exporta `build_agent`.")
    agent = module.build_agent()
    if hasattr(agent, "_max_iterations"):
        agent._max_iterations = max_iterations
    return agent


def run_scenario(
    scenario: Scenario,
    *,
    module_name: str,
    max_iterations: int,
) -> dict[str, Any]:
    """Ejecuta el agente sobre un escenario y devuelve un registro completo.

    El goal se comprueba sobre el estado del mundo (no sobre el texto del
    agente) con `check_goal`, para una métrica fiable. Nunca propaga
    excepciones: un fallo del agente se captura en el propio registro para
    que el resto del dataset siga corriendo.
    """
    world = copy.deepcopy(scenario.initial_world)
    optimal = OPTIMAL_CALLS.get(scenario.id)

    record: dict[str, Any] = {
        "scenario": scenario.id,
        "difficulty": scenario.difficulty,
        "description": scenario.description,
        "user_message": scenario.user_message,
        "goal": scenario.goal,
        "optimal_calls": optimal,
    }

    started = time.perf_counter()
    run_error: str | None = None
    result = None
    try:
        agent = _build_agent(module_name, max_iterations)
        for fn, schema in make_world_tools(world):
            agent.register_tool(fn, schema)
        result = agent.run(scenario.user_message)
    except Exception:  # noqa: BLE001 — se registra, no se propaga.
        run_error = traceback.format_exc()
    latency = time.perf_counter() - started

    achieved, reason = check_goal(world, scenario.goal)

    steps = [asdict(step) for step in result.steps] if result is not None else []
    num_tool_calls = len(steps)
    num_tool_errors = sum(1 for s in steps if s.get("error"))

    record.update(
        {
            "goal_achieved": achieved,
            "goal_reason": reason,
            "answer": result.answer if result is not None else None,
            "agent_error": result.error if result is not None else None,
            "run_error": run_error,
            "steps": steps,
            "num_tool_calls": num_tool_calls,
            "num_tool_errors": num_tool_errors,
            "calls_over_optimal": (
                num_tool_calls - optimal if optimal is not None else None
            ),
            "latency_seconds": round(latency, 3),
            "input_tokens": result.input_tokens if result is not None else None,
            "output_tokens": result.output_tokens if result is not None else None,
            "rubric": score_rubric(steps),
        }
    )
    
    # Diagnóstico: sólo tiene sentido sobre un caso fallido; en
    # los resueltos queda en None para no inventar una categoría.
    record["failure"] = classify_failure(record) if not achieved else None

    return record


# =============================================================================
# 6. Agregación de métricas
# =============================================================================
#
# Resume la lista de casos de una corrida en un único dict de métricas.
# Con `--repeat N` agrega además estabilidad por escenario, que es lo que
# distingue un fallo estructural de la varianza del modelo.


def build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega métricas cuantitativas por caso en un resumen del run."""
    total = len(cases)
    solved = sum(1 for c in cases if c["goal_achieved"])

    def _avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    latencies = [c["latency_seconds"] for c in cases if c["latency_seconds"] is not None]
    in_tokens = [c["input_tokens"] for c in cases if c["input_tokens"] is not None]
    out_tokens = [c["output_tokens"] for c in cases if c["output_tokens"] is not None]

    by_difficulty: dict[str, dict[str, int]] = {}
    for c in cases:
        bucket = by_difficulty.setdefault(
            c["difficulty"], {"total": 0, "solved": 0}
        )
        bucket["total"] += 1
        if c["goal_achieved"]:
            bucket["solved"] += 1

    # Desglose de fallos por categoría. El denominador son los
    # casos fallidos, no el total: un escenario resuelto no tiene causa
    # de fallo que contar. `infra_error` se cuenta acá pero NO es un fallo
    # del agente; hay que poder descontarlo al leer la tabla.
    failed = [c for c in cases if not c["goal_achieved"]]

    failures_by_category: dict[str, int] = {}
    for c in failed:
        failure = c.get("failure")
        if not failure:
            continue  # corrida anterior a la taxonomía, o caso sin diagnóstico.
        category = failure["category"]
        failures_by_category[category] = failures_by_category.get(category, 0) + 1

    # Más frecuente primero: es lo que el lector busca arriba en la tabla.
    failures_by_category = dict(
        sorted(failures_by_category.items(), key=lambda kv: kv[1], reverse=True)
    )

    rubrics = [c["rubric"] for c in cases if c.get("rubric")]
    rubric_dims = (
        "exploration_before_action",
        "no_redundant_actions",
        "no_hallucinated_ids",
        "error_recovery",
        "total",
        "normalized",
    )
    rubric_avg = {
        dim: _avg([r[dim] for r in rubrics]) for dim in rubric_dims
    }

    by_scenario: dict[str, dict[str, Any]] = {}
    for c in cases:
        b = by_scenario.setdefault(
            c["scenario"],
            {
                "difficulty": c["difficulty"],
                "runs": 0,
                "solved": 0,
                "_calls": [],
                "_rubric": [],
            },
        )
        b["runs"] += 1
        if c["goal_achieved"]:
            b["solved"] += 1
        b["_calls"].append(c["num_tool_calls"])
        if c.get("rubric"):
            b["_rubric"].append(c["rubric"]["total"])
    for b in by_scenario.values():
        b["success_rate"] = (
            round(b["solved"] / b["runs"], 3) if b["runs"] else None
        )
        b["avg_calls"] = _avg(b.pop("_calls"))
        b["avg_rubric_total"] = _avg(b.pop("_rubric"))

    return {
        "total_scenarios": total,
        "num_scenarios": len(by_scenario),
        "solved": solved,
        "success_rate": round(solved / total, 3) if total else None,
        "by_difficulty": by_difficulty,
        "by_scenario": by_scenario,
        "avg_latency_seconds": _avg(latencies),
        "total_input_tokens": sum(in_tokens) if in_tokens else None,
        "total_output_tokens": sum(out_tokens) if out_tokens else None,
        "rubric_avg": rubric_avg,
        "failures_total": len(failed),
        "failures_by_category": failures_by_category,
        "cases_with_run_error": sum(1 for c in cases if c["run_error"]),
    }


# =============================================================================
# 7. Reporte legible
# =============================================================================
#
# Traduce el resumen a Markdown. Es la única salida pensada para una
# persona: `summary.json` es para máquinas y `cases/` para inspección.


def _render_failure_section(
    summary: dict[str, Any], cases: list[dict[str, Any]]
) -> list[str]:
    """Arma la sección de análisis de errores del reporte.

    Extraída a un helper —a diferencia de las demás secciones, que se
    arman inline en `_render_report`— porque tiene lógica propia: un
    cruce de datos (agrupar escenarios por categoría) y dos ramas
    condicionales. Así también se puede testear sin renderizar el
    reporte entero.

    Devuelve las líneas Markdown; no escribe nada.
    """
    total_failures = summary.get("failures_total") or 0
    by_category = summary.get("failures_by_category") or {}

    lines: list[str] = []
    lines.append("## Análisis de errores — modos de fallo")
    lines.append("")

    if not total_failures:
        lines.append("Sin fallos en esta corrida: no hay modos de fallo que desglosar.")
        lines.append("")
        return lines

    # `failures_by_category` trae los conteos pero pierde el rastro de qué
    # escenario cayó en cada categoría: se reconstruye desde `cases`.
    scenarios_by_category: dict[str, list[str]] = {}
    failed_cases = [c for c in cases if not c["goal_achieved"] and c.get("failure")]
    for c in failed_cases:
        category = c["failure"]["category"]
        scenarios_by_category.setdefault(category, []).append(c["scenario"])

    lines.append(
        "Desglose de los fallos por causa. Los porcentajes son sobre el total "
        f"de **fallos** ({total_failures}), no sobre el total de casos: un "
        "escenario resuelto no tiene causa de fallo que contar."
    )
    lines.append("")
    lines.append("| Categoría | Casos | % de fallos | Escenarios |")
    lines.append("|---|---|---|---|")
    for category, count in by_category.items():
        scenarios = ", ".join(sorted(set(scenarios_by_category.get(category, []))))
        lines.append(
            f"| {category} | {count} | {count / total_failures:.0%} | "
            f"{scenarios or '—'} |"
        )
    lines.append("")

    if "infra_error" in by_category:
        lines.append(
            f"> **Nota.** {by_category['infra_error']} caso(s) son `infra_error`: "
            "la corrida falló antes de poder evaluar al agente (p. ej. credenciales "
            "vencidas). No son fallos del agente y hay que descontarlos al leer "
            "esta tabla."
        )
        lines.append("")

    lines.append("### Detalle por caso")
    lines.append("")
    lines.append("| Escenario | Dif. | Categoría | Evidencia |")
    lines.append("|---|---|---|---|")
    for c in failed_cases:
        failure = c["failure"]
        lines.append(
            f"| {c['scenario']} | {c['difficulty']} | {failure['category']} | "
            f"{failure['evidence']} |"
        )
    lines.append("")
    return lines


def _render_report(meta: dict[str, Any], summary: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    """Genera un resumen legible por humanos en Markdown."""
    lines: list[str] = []
    lines.append("# Reporte de evaluación M3")
    lines.append("")
    lines.append(f"- Fecha (UTC): {meta['timestamp']}")
    lines.append(f"- Módulo del agente: `{meta['module']}`")
    lines.append(f"- Proveedor LLM: {meta['llm_provider']}")
    lines.append(f"- max_iterations: {meta['max_iterations']}")
    if meta.get("repeat", 1) > 1:
        lines.append(f"- Repeticiones por escenario: {meta['repeat']}")
    lines.append("")
    lines.append("## Resumen")
    lines.append("")
    if meta.get("repeat", 1) > 1:
        lines.append(
            f"- Corridas exitosas: **{summary['solved']}/{summary['total_scenarios']}** "
            f"(tasa media {summary['success_rate']}, sobre "
            f"{summary['num_scenarios']} escenarios × {meta['repeat']} repeticiones)"
        )
    else:
        lines.append(
            f"- Escenarios resueltos: **{summary['solved']}/{summary['total_scenarios']}** "
            f"(éxito {summary['success_rate']})"
        )
    lines.append(f"- Latencia promedio: {summary['avg_latency_seconds']} s")
    lines.append(
        f"- Tokens totales: {summary['total_input_tokens']} in / "
        f"{summary['total_output_tokens']} out"
    )
    lines.append(f"- Casos con error de ejecución: {summary['cases_with_run_error']}")
    lines.append("")
    lines.append("### Por dificultad")
    lines.append("")
    lines.append("| Dificultad | Resueltos |")
    lines.append("|---|---|")
    for diff, bucket in summary["by_difficulty"].items():
        lines.append(f"| {diff} | {bucket['solved']}/{bucket['total']} |")
    lines.append("")
    if meta.get("repeat", 1) > 1:
        lines.append("## Estabilidad por escenario (repeticiones)")
        lines.append("")
        lines.append(
            "| Escenario | Dif. | Éxito (k/N) | Tasa | Calls medio | Rúbrica media |"
        )
        lines.append("|---|---|---|---|---|---|")
        for sid, b in summary["by_scenario"].items():
            lines.append(
                f"| {sid} | {b['difficulty']} | {b['solved']}/{b['runs']} | "
                f"{b['success_rate']} | {b['avg_calls']} | {b['avg_rubric_total']} |"
            )
        lines.append("")
    else:
        lines.append("## Detalle por escenario")
        lines.append("")
        lines.append(
            "| Escenario | Dif. | Goal | Calls | Óptimo | Errores | Latencia (s) |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for c in cases:
            goal_mark = "✅" if c["goal_achieved"] else "❌"
            optimal = c["optimal_calls"] if c["optimal_calls"] is not None else "—"
            lines.append(
                f"| {c['scenario']} | {c['difficulty']} | {goal_mark} | "
                f"{c['num_tool_calls']} | {optimal} | {c['num_tool_errors']} | "
                f"{c['latency_seconds']} |"
            )
        lines.append("")
    lines.append("## Calidad de proceso — rúbrica determinista (0–8)")
    lines.append("")
    lines.append(
        "Métrica cualitativa: puntúa la traza de acciones, "
        "no el texto. Dimensiones 0–2: **Explor.** (exploró antes de actuar), "
        "**No-red.** (sin acciones repetidas), **No-halu.** (sin ids "
        "inexistentes), **Recup.** (cambió de acción tras un error)."
    )
    lines.append("")
    lines.append(
        "| Escenario | Explor. | No-red. | No-halu. | Recup. | Total | Norm. |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    if meta.get("repeat", 1) == 1:
        for c in cases:
            r = c.get("rubric") or {}
            lines.append(
                f"| {c['scenario']} | {r.get('exploration_before_action')} | "
                f"{r.get('no_redundant_actions')} | {r.get('no_hallucinated_ids')} | "
                f"{r.get('error_recovery')} | {r.get('total')}/{r.get('max')} | "
                f"{r.get('normalized')} |"
            )
    ra = summary.get("rubric_avg") or {}
    lines.append(
        f"| **promedio** | {ra.get('exploration_before_action')} | "
        f"{ra.get('no_redundant_actions')} | {ra.get('no_hallucinated_ids')} | "
        f"{ra.get('error_recovery')} | {ra.get('total')} | {ra.get('normalized')} |"
    )
    lines.append("")
    lines.extend(_render_failure_section(summary, cases))
    return "\n".join(lines)


def _llm_provider_label() -> str:
    """Describe el proveedor LLM activo según el entorno (solo informativo)."""
    import os

    if os.environ.get("OLLAMA_HOST"):
        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        return f"Ollama ({model})"
    if os.environ.get("BEDROCK_MODEL_ID"):
        region = os.environ.get("AWS_REGION", "us-east-1")
        return f"Bedrock ({os.environ['BEDROCK_MODEL_ID']}, {region})"
    return "no configurado"


# =============================================================================
# 8. Orquestación y CLI
# =============================================================================
#
# Ata todo: recorre el dataset, persiste los tres artefactos y expone los
# modos de uso. `reclassify_run` es la excepción que no llama al LLM:
# rediagnostica trazas ya guardadas.


def _case_filename(scenario_id: str, repeat: int, index: int) -> str:
    """Nombre del registro de un caso; con repeticiones lleva sufijo `__rN`."""
    return f"{scenario_id}.json" if repeat == 1 else f"{scenario_id}__r{index + 1}.json"


def reclassify_run(run_dir: Path) -> int:
    """Reclasifica los fallos de una corrida ya ejecutada, sin llamar al LLM.

    Relee `summary.json`, vuelve a pasar `classify_failure` sobre cada caso
    y reescribe los tres artefactos (`cases/*.json`, `summary.json` y
    `report.md`). Permite ajustar la taxonomía o sus umbrales y ver el
    efecto sobre una corrida real en segundos, sin volver a gastar tokens:
    las trazas ya están en disco.
    """
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"No se encontró {summary_path}.")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = payload["meta"]
    cases = payload["cases"]
    repeat = meta.get("repeat", 1)
    cases_dir = run_dir / "cases"

    changed = 0
    for case in cases:
        before = (case.get("failure") or {}).get("category")
        case["failure"] = (
            classify_failure(case) if not case["goal_achieved"] else None
        )
        if before != (case.get("failure") or {}).get("category"):
            changed += 1
        fname = _case_filename(case["scenario"], repeat, case.get("repeat_index", 0))
        (cases_dir / fname).write_text(
            json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary = build_summary(cases)
    summary_path.write_text(
        json.dumps(
            {"meta": meta, "summary": summary, "cases": cases},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        _render_report(meta, summary, cases), encoding="utf-8"
    )

    print(
        f"# Reclasificados {len(cases)} caso(s); "
        f"{changed} cambiaron de categoría.",
        file=sys.stderr,
    )
    print(f"# Reporte: {run_dir / 'report.md'}", file=sys.stderr)
    return 0


def run_all(
    *,
    scenario_spec: str | None,
    module_name: str,
    scenarios_dir: Path,
    out_dir: Path,
    max_iterations: int,
    repeat: int = 1,
) -> int:
    """Corre la evaluación completa y persiste los registros. Devuelve exit code."""
    scenarios = _resolve_scenarios(scenario_spec, scenarios_dir)
    if not scenarios:
        print(f"(sin escenarios en {scenarios_dir})", file=sys.stderr)
        return 1

    # Cargar el `.env` (Ollama/Bedrock) antes de etiquetar el proveedor:
    # `from_env()` lo hace recién al construir el agente, así que sin esto
    # el label reportaría "no configurado" aunque el `.env` sí lo defina.
    load_env_files()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / timestamp
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": timestamp,
        "module": module_name,
        "llm_provider": _llm_provider_label(),
        "max_iterations": max_iterations,
        "repeat": repeat,
        "python": platform.python_version(),
        "scenarios_dir": str(scenarios_dir),
    }

    total_label = f"{len(scenarios)} escenario(s)"
    if repeat > 1:
        total_label += f" × {repeat} repeticiones"
    print(f"# Evaluación M3 — {total_label}", file=sys.stderr)
    print(f"# Proveedor LLM: {meta['llm_provider']}", file=sys.stderr)
    print(f"# Registros en: {run_dir}", file=sys.stderr)
    print(file=sys.stderr)

    cases: list[dict[str, Any]] = []
    for scenario in scenarios:
        for r in range(repeat):
            label = (
                scenario.id
                if repeat == 1
                else f"{scenario.id} #{r + 1}/{repeat}"
            )
            print(
                f"→ {label} ({scenario.difficulty}) ...",
                end="",
                flush=True,
                file=sys.stderr,
            )
            case = run_scenario(
                scenario,
                module_name=module_name,
                max_iterations=max_iterations,
            )
            case["repeat_index"] = r
            cases.append(case)

            fname = _case_filename(scenario.id, repeat, r)
            (cases_dir / fname).write_text(
                json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            status = "OK " if case["goal_achieved"] else "FALLÓ"
            extra = " (error de ejecución)" if case["run_error"] else ""
            print(
                f" {status} — {case['num_tool_calls']} calls, "
                f"{case['latency_seconds']}s{extra}",
                file=sys.stderr,
            )

    summary = build_summary(cases)
    payload = {"meta": meta, "summary": summary, "cases": cases}
    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(
        _render_report(meta, summary, cases), encoding="utf-8"
    )

    print(file=sys.stderr)
    print(
        f"# Resuelto {summary['solved']}/{summary['total_scenarios']} "
        f"(éxito {summary['success_rate']})",
        file=sys.stderr,
    )
    print(f"# Resumen: {run_dir / 'summary.json'}", file=sys.stderr)
    print(f"# Reporte: {run_dir / 'report.md'}", file=sys.stderr)

    # Éxito de proceso: todo el dataset corrió sin errores de ejecución.
    # El éxito por escenario (goal) se reporta en el resumen, no en el exit.
    return 0 if summary["cases_with_run_error"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="student_framework/eval/run.py",
        description="Ejecuta el agente sobre el dataset de escenarios de M3.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help=(
            "Escenario a evaluar (id, dificultad o path al JSON). "
            "Por defecto: todos."
        ),
    )
    parser.add_argument(
        "--module",
        default=DEFAULT_MODULE,
        help=f"Módulo que expone `build_agent` (por defecto: {DEFAULT_MODULE}).",
    )
    parser.add_argument(
        "--scenarios-dir",
        default=str(DEFAULT_SCENARIOS_DIR),
        help="Directorio con los escenarios JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directorio raíz donde escribir los registros del run.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=(
            "Tope de iteraciones del bucle del agente "
            f"(por defecto: {DEFAULT_MAX_ITERATIONS})."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Repeticiones por escenario para promediar la varianza del LLM "
            "(por defecto: 1). Con N>1 el reporte agrega tasa de éxito media "
            "y una tabla de estabilidad por escenario."
        ),
    )
    parser.add_argument(
        "--reclassify",
        default=None,
        metavar="RUN_DIR",
        help=(
            "Reclasifica los fallos de una corrida ya ejecutada y reescribe "
            "sus artefactos, sin llamar al LLM. Útil tras ajustar la "
            "taxonomía de modos de fallo o sus umbrales."
        ),
    )
    args = parser.parse_args(argv)

    if args.reclassify:
        return reclassify_run(Path(args.reclassify))

    if args.repeat < 1:
        parser.error("--repeat debe ser >= 1")

    return run_all(
        scenario_spec=args.scenario,
        module_name=args.module,
        scenarios_dir=Path(args.scenarios_dir),
        out_dir=Path(args.out_dir),
        max_iterations=args.max_iterations,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    sys.exit(main())
