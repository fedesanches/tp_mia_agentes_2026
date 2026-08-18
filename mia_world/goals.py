"""Comprobación de la condición de victoria de un escenario.

`check_goal` inspecciona el estado del `World` (no el texto del agente).
Esto da una métrica fiable: para que un escenario cuente como resuelto
hace falta que el mundo haya cambiado, no que el agente diga que lo ha
resuelto.

Tipos de goal soportados
------------------------

Hojas (condiciones atómicas sobre el estado):
  - `item_open`        — el `Item` indicado tiene `open_state == "open"`.
  - `agent_in_room`    — `world.current_room == goal["room"]`.
  - `item_in_inventory`— `goal["item"]` está en `world.inventory`.

Combinadores (composición de sub-goals, recursivos):
  - `all_of`   — todos los sub-goals se cumplen (AND).
  - `any_of`   — al menos un sub-goal se cumple (OR).
  - `sequence` — todos los sub-goals se cumplen **y** sus eventos
                 asociados ocurrieron en el orden listado. El orden se
                 verifica como subsecuencia de `world.event_log`, que las
                 herramientas pueblan ("enter:", "take:", "open:"). Solo
                 admite sub-goals de tipo hoja.

Forma de un goal compuesto::

    {"type": "all_of", "goals": [
        {"type": "item_in_inventory", "item": "documento"},
        {"type": "item_open", "item": "puerta_principal"}
    ]}
"""

from __future__ import annotations

from typing import Any

from .state import World


def _goal_event(goal: dict[str, Any]) -> str:
    """Evento de `event_log` que corresponde a un sub-goal hoja.

    Usado por `sequence` para comprobar el orden temporal.
    """
    goal_type = goal.get("type")
    if goal_type == "item_open":
        return f"open:{goal['item']}"
    if goal_type == "item_in_inventory":
        return f"take:{goal['item']}"
    if goal_type == "agent_in_room":
        return f"enter:{goal['room']}"
    raise ValueError(
        f"`sequence` solo admite sub-goals hoja "
        f"(item_open / item_in_inventory / agent_in_room), no {goal_type!r}."
    )


def _check_sequence(
    world: World, goals: list[dict[str, Any]]
) -> tuple[bool, str]:
    sub = [check_goal(world, g) for g in goals]
    if not all(ok for ok, _ in sub):
        pending = "; ".join(reason for ok, reason in sub if not ok)
        return False, f"faltan condiciones: {pending}"

    pos = -1
    for goal in goals:
        event = _goal_event(goal)
        try:
            pos = world.event_log.index(event, pos + 1)
        except ValueError:
            return False, f"el evento {event!r} no ocurrió en el orden requerido"
    return True, "secuencia completada en el orden correcto"


def check_goal(world: World, goal: dict[str, Any]) -> tuple[bool, str]:
    goal_type = goal.get("type")
    if goal_type == "item_open":
        item_id = goal["item"]
        item = world.items.get(item_id)
        if item is None:
            return False, f"no existe el item {item_id!r}"
        ok = item.open_state == "open"
        return ok, f"{item.name} está {'abierta' if ok else 'cerrada'}"
    if goal_type == "agent_in_room":
        room_id = goal["room"]
        room_name = world.rooms[room_id].name if room_id in world.rooms else room_id
        ok = world.current_room == room_id
        return ok, f"{'estás' if ok else 'no estás'} en {room_name}"
    if goal_type == "item_in_inventory":
        item_id = goal["item"]
        item_name = world.items[item_id].name if item_id in world.items else item_id
        ok = item_id in world.inventory
        return ok, f"{item_name} {'está' if ok else 'no está'} en el inventario"
    if goal_type == "all_of":
        results = [check_goal(world, g) for g in goal["goals"]]
        ok = all(r[0] for r in results)
        return ok, "; ".join(r[1] for r in results)
    if goal_type == "any_of":
        results = [check_goal(world, g) for g in goal["goals"]]
        ok = any(r[0] for r in results)
        return ok, " | ".join(r[1] for r in results)
    if goal_type == "sequence":
        return _check_sequence(world, goal["goals"])
    raise ValueError(f"Tipo de goal desconocido: {goal_type!r}")
