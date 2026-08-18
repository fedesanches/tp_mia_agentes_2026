"""mia_world: mundo simulado fijo para el problema de M3.

Distribuido por el equipo docente. **No editar**. Los estudiantes
registran las herramientas devueltas por `make_world_tools(world)` en su
agente y resuelven los escenarios definidos en `scaffold/scenarios/`.

API pública:
  - `Item`, `Room`, `World`, `Scenario` — dataclasses del modelo de estado.
  - `load_scenario(path)` — parsea un escenario JSON.
  - `make_world_tools(world)` — devuelve los 4 verbos enlazados a `world`,
    cada uno como par `(callable, ToolSchema)` con esquema vía
    `ToolSchema.from_callable`.
  - `check_goal(world, goal_spec)` — comprueba si el mundo cumple la
    condición de victoria del escenario.
"""

from __future__ import annotations

from mia_world.goals import check_goal
from mia_world.scenarios import list_scenarios, load_scenario
from mia_world.state import Item, Room, Scenario, World
from mia_world.tools import make_world_tools

__all__ = [
    "Item",
    "Room",
    "Scenario",
    "World",
    "check_goal",
    "list_scenarios",
    "load_scenario",
    "make_world_tools",
]
