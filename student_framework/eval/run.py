"""Arnés de evaluación reproducible para M3 (US-02).

Ejecuta el agente del grupo sobre **todo** el dataset de escenarios del
mundo simulado sin pasos manuales, capturando por caso: el mensaje de
entrada, la respuesta final, cada llamada a herramienta (input/output/
error) y cualquier fallo del agente. Deja un registro por caso en disco y
un resumen agregado con métricas cuantitativas.

Uso típico (sin argumentos: corre los 8 escenarios):

    python student_framework/eval/run.py

Opciones útiles:

    python student_framework/eval/run.py --scenario easy      # un escenario
    python student_framework/eval/run.py --max-iterations 40  # tope de pasos
    python student_framework/eval/run.py --repeat 3           # promediar N corridas
    python student_framework/eval/run.py --out-dir <dir>      # dónde guardar

Salidas (bajo `student_framework/eval/runs/<timestamp>/`):
  - `cases/<scenario_id>.json`  — un registro completo por escenario.
  - `summary.json`              — métricas agregadas legibles por máquina.
  - `report.md`                 — resumen legible por humanos.

Requisitos: el módulo del agente (por defecto `student_framework`) debe
exportar `build_agent`, y debe haber un proveedor LLM configurado
(Bedrock con `BEDROCK_MODEL_ID`, u Ollama con `OLLAMA_HOST`). Ver README.
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
    permitir escenarios de horizonte largo y los experimentos de US-05.
    """
    module = importlib.import_module(module_name)
    if not hasattr(module, "build_agent"):
        raise SystemExit(f"El módulo {module_name!r} no exporta `build_agent`.")
    agent = module.build_agent()
    if hasattr(agent, "_max_iterations"):
        agent._max_iterations = max_iterations
    return agent


# --- Rúbrica cualitativa determinista (US-03, opción A) ------------------------
#
# Puntúa la *calidad del proceso* de resolución a partir de la traza de
# `steps` (no del texto libre), en 4 dimensiones de 0–2 (total 0–8). Es
# determinista y sin coste de LLM, por lo que es 100% reproducible: dos
# corridas con la misma traza dan el mismo puntaje.

RUBRIC_MAX = 8
_EXPLORE_TOOLS = {"look", "examine"}
_ACTION_TOOLS = {"use", "take"}


def _call_identity(step: dict[str, Any]) -> tuple[str | None, str]:
    """Identidad normalizada de una llamada, para detectar repeticiones."""
    name = step.get("tool_name")
    raw = step.get("tool_input")
    try:
        args = json.loads(raw) if raw else {}
        norm = json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        norm = raw or ""
    return (name, norm)


def _is_error_output(step: dict[str, Any]) -> bool:
    """True si el paso falló: error del bucle o tool_output `Error: ...`."""
    if step.get("error"):
        return True
    out = (step.get("tool_output") or "").strip().lower()
    return out.startswith("error:")


def _is_missing_id_output(step: dict[str, Any]) -> bool:
    """True si el tool_output indica un id inexistente o no visible."""
    out = (step.get("tool_output") or "").lower()
    return "no existe ning" in out or "no ves ning" in out


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
    return record


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
        "cases_with_run_error": sum(1 for c in cases if c["run_error"]),
    }


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
        "Métrica cualitativa (US-03, opción A): puntúa la traza de acciones, "
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

            fname = (
                f"{scenario.id}.json"
                if repeat == 1
                else f"{scenario.id}__r{r + 1}.json"
            )
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
    args = parser.parse_args(argv)

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
