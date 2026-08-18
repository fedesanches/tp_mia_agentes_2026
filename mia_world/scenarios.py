"""Carga de escenarios desde JSON.

Formato esperado (ver `scenarios/01-study-with-key.json` para un ejemplo
completo):

    {
      "id": "study-with-key",
      "description": "...",
      "user_message": "Mensaje inicial que el agente recibe en run()",
      "start_room": "estudio",
      "rooms": {
        "estudio": { "name": "...", "description": "...", "items": [...] }
      },
      "items": {
        "alfombra": { "name": "...", "description": "...",
                       "container": true, "contains": ["llave_oro"] },
        ...
      },
      "goal": { "type": "item_open", "item": "puerta_principal" }
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import Item, Room, Scenario, World


def _item_from_spec(item_id: str, spec: dict[str, Any]) -> Item:
    return Item(
        id=item_id,
        name=spec["name"],
        description=spec["description"],
        takeable=spec.get("takeable", False),
        container=spec.get("container", False),
        contains=list(spec.get("contains", [])),
        hidden_by=spec.get("hidden_by"),
        locked=spec.get("locked"),
        open_state=spec.get("open_state", "closed"),
        inserted=list(spec.get("inserted", [])),
    )


def _room_from_spec(room_id: str, spec: dict[str, Any]) -> Room:
    return Room(
        id=room_id,
        name=spec["name"],
        description=spec["description"],
        items=list(spec.get("items", [])),
        exits=dict(spec.get("exits", {})),
        locked_exits=dict(spec.get("locked_exits", {})),
    )


def _goal_refs(goal: dict[str, Any]) -> list[tuple[str, str]]:
    """Referencias (kind, id) que un goal hace a items/salas, recursivamente."""
    goal_type = goal.get("type")
    if goal_type in ("item_open", "item_in_inventory"):
        return [("item", goal["item"])]
    if goal_type == "agent_in_room":
        return [("room", goal["room"])]
    if goal_type in ("all_of", "any_of", "sequence"):
        refs: list[tuple[str, str]] = []
        for sub in goal.get("goals", []):
            refs.extend(_goal_refs(sub))
        return refs
    return []


def _validate_world(
    rooms: dict[str, Room],
    items: dict[str, Item],
    goal: dict[str, Any],
    scenario_id: str,
) -> None:
    """Chequea integridad referencial y falla en carga, no en mitad de un run.

    Sin esto, un id mal escrito aflora tarde: como un `KeyError` en una
    herramienta o como comportamiento silenciosamente incorrecto (p. ej. una
    salida gated que se vuelve transitable).
    """
    problems: list[str] = []

    def require_item(ref: str, where: str) -> None:
        if ref not in items:
            problems.append(f"{where} -> item inexistente {ref!r}")

    for rid, room in rooms.items():
        for iid in room.items:
            require_item(iid, f"room {rid}.items")
        for direction, dest in room.exits.items():
            if dest not in rooms:
                problems.append(f"room {rid}.exits[{direction!r}] -> sala inexistente {dest!r}")
        for direction, gate_id in room.locked_exits.items():
            require_item(gate_id, f"room {rid}.locked_exits[{direction!r}]")
            if direction not in room.exits:
                problems.append(
                    f"room {rid}.locked_exits[{direction!r}] no tiene una salida correspondiente en `exits`"
                )

    for iid, item in items.items():
        for child in item.contains:
            require_item(child, f"item {iid}.contains")
        if item.hidden_by is not None:
            require_item(item.hidden_by, f"item {iid}.hidden_by")
        locked = item.locked or {}
        if locked.get("requires_item"):
            require_item(locked["requires_item"], f"item {iid}.locked.requires_item")
        for req in locked.get("requires_items", []):
            require_item(req, f"item {iid}.locked.requires_items")

    for kind, ref in _goal_refs(goal):
        if kind == "item" and ref not in items:
            problems.append(f"goal -> item inexistente {ref!r}")
        if kind == "room" and ref not in rooms:
            problems.append(f"goal -> sala inexistente {ref!r}")

    if problems:
        joined = "\n  - ".join(problems)
        raise ValueError(
            f"Escenario {scenario_id!r} mal formado (integridad referencial):\n  - {joined}"
        )


def load_scenario(path: str | Path) -> Scenario:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = {iid: _item_from_spec(iid, spec) for iid, spec in data["items"].items()}
    rooms = {rid: _room_from_spec(rid, spec) for rid, spec in data["rooms"].items()}
    start_room = data["start_room"]
    if start_room not in rooms:
        raise ValueError(
            f"start_room {start_room!r} no aparece en `rooms` del escenario {data['id']!r}."
        )
    _validate_world(rooms, items, data["goal"], data["id"])
    world = World(rooms=rooms, items=items, current_room=start_room)
    return Scenario(
        id=data["id"],
        description=data["description"],
        user_message=data["user_message"],
        initial_world=world,
        goal=data["goal"],
        difficulty=data.get("difficulty", "unspecified"),
    )


def list_scenarios(directory: str | Path) -> list[Scenario]:
    """Carga todos los escenarios `.json` de un directorio, ordenados por nombre de fichero.

    No filtra por dificultad — el llamante puede filtrar inspeccionando
    `scenario.difficulty`. Ordenar por nombre de fichero coincide con la
    convención `01-...`, `02-...`, `03-...` que da una progresión de
    dificultad creciente.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe el directorio de escenarios: {directory}")
    return [load_scenario(p) for p in sorted(directory.glob("*.json"))]
