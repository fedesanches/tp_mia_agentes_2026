"""Runner por línea de comandos para los escenarios de M3.

Uso típico:

    # Listar escenarios disponibles
    python -m mia_world.cli list

    # Ejecutar un escenario por id (busca en ./scenarios) o por path
    python -m mia_world.cli run --scenario easy
    python -m mia_world.cli run --scenario color-locks
    python -m mia_world.cli run --scenario scenarios/03-hard-library-search.json

`run` importa `build_agent` del módulo indicado (por defecto
`student_framework`), registra las 4 herramientas del mundo en el agente
y le pasa el `user_message` del escenario. Al terminar, comprueba la
condición de victoria con `check_goal` y la imprime junto con los pasos.

Esto requiere que el módulo del agente exporte `build_agent` (ya
necesario para los tests de conformidad) y que haya configurada una
proveedor LLM (por defecto, Bedrock con `BEDROCK_MODEL_ID`).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mia_world.goals import check_goal
from mia_world.scenarios import list_scenarios, load_scenario
from mia_world.state import Scenario
from mia_world.tools import make_world_tools


DEFAULT_SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"


def _resolve_scenario(spec: str, scenarios_dir: Path) -> Scenario:
    """Resuelve `spec` a un `Scenario`. Acepta path, id o difficulty."""
    path = Path(spec)
    if path.is_file():
        return load_scenario(path)

    available = list_scenarios(scenarios_dir)

    by_id = {sc.id: sc for sc in available}
    if spec in by_id:
        return by_id[spec]

    # Puede haber varios escenarios por dificultad. `available` viene
    # ordenado por nombre de fichero (convención `NN-...`), así que el
    # primero es el de menor índice. Para elegir uno concreto, usá su id.
    by_diff = [sc for sc in available if sc.difficulty == spec]
    if by_diff:
        return by_diff[0]

    options = ", ".join(sorted(sc.id for sc in available)) or "(ninguno)"
    raise SystemExit(
        f"No se encontró el escenario {spec!r}. Disponibles: {options}."
    )


def _cmd_list(args: argparse.Namespace) -> int:
    scenarios_dir = Path(args.scenarios_dir)
    scenarios = list_scenarios(scenarios_dir)
    if not scenarios:
        print(f"(sin escenarios en {scenarios_dir})")
        return 0
    print(f"Escenarios en {scenarios_dir}:")
    print()
    width_id = max(len(sc.id) for sc in scenarios)
    width_diff = max(len(sc.difficulty) for sc in scenarios)
    for sc in scenarios:
        print(
            f"  {sc.difficulty:<{width_diff}}  {sc.id:<{width_id}}  {sc.description}"
        )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    scenario = _resolve_scenario(args.scenario, Path(args.scenarios_dir))
    world = scenario.initial_world

    module = importlib.import_module(args.module)
    if not hasattr(module, "build_agent"):
        raise SystemExit(
            f"El módulo {args.module!r} no exporta `build_agent`."
        )
    agent = module.build_agent()
    for fn, schema in make_world_tools(world):
        agent.register_tool(fn, schema)

    print(f"# Escenario: {scenario.id} ({scenario.difficulty})", file=sys.stderr)
    print(f"# {scenario.description}", file=sys.stderr)
    print(file=sys.stderr)

    result = agent.run(scenario.user_message)
    achieved, reason = check_goal(world, scenario.goal)

    output: dict[str, Any] = {
        "scenario": scenario.id,
        "difficulty": scenario.difficulty,
        "goal": scenario.goal,
        "goal_achieved": achieved,
        "goal_reason": reason,
        "agent_result": asdict(result),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if achieved else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mia_world")
    parser.add_argument(
        "--scenarios-dir",
        default=str(DEFAULT_SCENARIOS_DIR),
        help="Directorio donde buscar escenarios (por defecto: scaffold/scenarios).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list", help="Lista los escenarios disponibles.")
    list_p.set_defaults(func=_cmd_list)

    run_p = sub.add_parser(
        "run",
        help="Ejecuta un escenario con el agente del módulo indicado.",
    )
    run_p.add_argument(
        "--scenario",
        required=True,
        help="Identificador del escenario (id, dificultad o path al JSON).",
    )
    run_p.add_argument(
        "--module",
        default="student_framework",
        help="Módulo Python que expone `build_agent`. Por defecto: student_framework.",
    )
    run_p.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
