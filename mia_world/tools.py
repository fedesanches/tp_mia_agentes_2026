"""Herramientas genéricas del mundo simulado (M3).

Verbos base: `look`, `examine`, `take`, `use`. En escenarios con varias
salas se añade un quinto verbo, `go`, para navegar entre salas. Cada uno
se construye con `make_world_tools(world)`, que devuelve pares
`(callable, ToolSchema)` listos para `agent.register_tool(...)`.

`go` solo se registra cuando alguna sala del mundo define `exits`; los
escenarios de una sola sala mantienen exactamente los cuatro verbos
clásicos (retrocompatible).

Los esquemas se generan con `ToolSchema.from_callable` a partir de las
funciones de referencia `_*_tool` (firma + docstring + `Annotated`).
"""

from __future__ import annotations

from typing import Annotated, Callable

from pydantic import Field

from mia_agents.types import ToolSchema

from .state import World


def _format_item_line(world: World, item_id: str) -> str:
    item = world.items[item_id]
    suffix = ""
    if item.locked is not None:
        suffix = f" ({'abierta' if item.open_state == 'open' else 'cerrada'})"
    return f"  - {item.name} [id: {item.id}]{suffix}"


def _visible_in_room(world: World) -> list[str]:
    """Items visibles desde la sala actual."""
    visible: list[str] = []
    room = world.rooms[world.current_room]
    for item_id in room.items:
        if world.is_visible(item_id):
            visible.append(item_id)
    for parent_id in room.items:
        parent = world.items.get(parent_id)
        if parent is None or not parent.container or parent_id not in world.revealed:
            continue
        for child_id in parent.contains:
            if child_id not in visible and world.is_visible(child_id):
                visible.append(child_id)
    return visible


# --- Firmas de referencia (esquema para el LLM) --------------------------------


def look() -> str:
    """Describe la sala actual: items visibles (con sus ids) y el inventario."""
    raise NotImplementedError("usar make_world_tools(world)")


def examine(
    target: Annotated[
        str,
        Field(description="Id del objeto a examinar (p. ej. 'alfombra')."),
    ],
) -> str:
    """Examina un objeto visible o del inventario; si es contenedor, revela su contenido."""
    raise NotImplementedError("usar make_world_tools(world)")


def take(
    item: Annotated[str, Field(description="Id del objeto a tomar.")],
) -> str:
    """Toma un objeto visible y lo añade al inventario."""
    raise NotImplementedError("usar make_world_tools(world)")


def use(
    item: Annotated[
        str,
        Field(description="Id del objeto del inventario que querés usar."),
    ],
    target: Annotated[
        str,
        Field(description="Id del objeto de la sala sobre el que aplicarlo."),
    ],
) -> str:
    """Usa un objeto del inventario sobre un objeto visible (p. ej. llave en puerta)."""
    raise NotImplementedError("usar make_world_tools(world)")


def go(
    direction: Annotated[
        str,
        Field(
            description=(
                "Dirección de la salida a tomar desde la sala actual "
                "(p. ej. 'norte'). Usá `look` para ver las salidas disponibles."
            )
        ),
    ],
) -> str:
    """Te desplazas por una salida de la sala actual hacia la sala contigua."""
    raise NotImplementedError("usar make_world_tools(world)")


look_schema = ToolSchema.from_callable(look)
examine_schema = ToolSchema.from_callable(examine)
take_schema = ToolSchema.from_callable(take)
use_schema = ToolSchema.from_callable(use)
go_schema = ToolSchema.from_callable(go)


# --- Implementaciones enlazadas a World ----------------------------------------


def _make_look(world: World) -> Callable[[], str]:
    def look_impl() -> str:
        room = world.rooms[world.current_room]
        lines = [f"Estás en {room.name}.", room.description]
        visible = _visible_in_room(world)
        if visible:
            lines.append("Ves:")
            for item_id in visible:
                lines.append(_format_item_line(world, item_id))
        else:
            lines.append("No ves nada de interés.")
        if room.exits:
            parts = []
            for direction in sorted(room.exits.keys()):
                gate_id = room.locked_exits.get(direction)
                gate = world.items.get(gate_id) if gate_id else None
                if gate is not None and gate.open_state != "open":
                    parts.append(f"{direction} (bloqueada por {gate.name})")
                else:
                    parts.append(direction)
            lines.append("Salidas: " + ", ".join(parts) + ".")
        if world.inventory:
            inv_names = ", ".join(
                f"{world.items[i].name} [id: {i}]" for i in world.inventory
            )
            lines.append(f"Llevas: {inv_names}.")
        return "\n".join(lines)

    return look_impl


def _make_examine(world: World) -> Callable[..., str]:
    def examine_impl(target: str) -> str:
        if target not in world.items:
            return f"Error: no existe ningún objeto con id {target!r}."
        if not world.is_visible(target) and target not in world.inventory:
            return f"Error: no ves ningún {target!r} aquí."
        item = world.items[target]
        lines = [f"{item.name}: {item.description}"]
        is_locked_closed = item.locked is not None and item.open_state != "open"
        if item.container:
            if is_locked_closed:
                lines.append(
                    "Está cerrado con llave. No puedes ver el interior hasta abrirlo."
                )
            else:
                world.revealed.add(target)
                child_lines = [
                    f"  - {world.items[child_id].name} [id: {child_id}]"
                    for child_id in item.contains
                    if child_id in world.items
                ]
                if child_lines:
                    lines.append("Contiene:")
                    lines.extend(child_lines)
                else:
                    lines.append("Está vacío.")
        elif item.locked is not None:
            state = "abierta" if item.open_state == "open" else "cerrada"
            lines.append(f"Estado: {state}.")
            required_items = item.locked.get("requires_items")
            if required_items and item.open_state != "open":
                if item.inserted:
                    done = ", ".join(
                        world.items[i].name if i in world.items else i
                        for i in item.inserted
                    )
                    lines.append(f"Piezas colocadas: {done}.")
                missing = [r for r in required_items if r not in item.inserted]
                missing_names = ", ".join(
                    world.items[m].name if m in world.items else m for m in missing
                )
                lines.append(
                    f"Requiere {len(required_items)} piezas en total; "
                    f"faltan: {missing_names}."
                )
        return "\n".join(lines)

    return examine_impl


def _make_take(world: World) -> Callable[..., str]:
    def take_impl(item: str) -> str:
        if item not in world.items:
            return f"Error: no existe ningún objeto con id {item!r}."
        obj = world.items[item]
        if item in world.inventory:
            return f"Error: ya llevas {obj.name}."
        if not world.is_visible(item):
            return f"Error: {item!r} no es visible o accesible desde aquí."
        if not obj.takeable:
            return f"Error: {obj.name} no es algo que puedas llevarte."
        if obj.hidden_by:
            parent = world.items.get(obj.hidden_by)
            if parent is not None and item in parent.contains:
                parent.contains.remove(item)
        else:
            room = world.rooms[world.current_room]
            if item in room.items:
                room.items.remove(item)
        world.inventory.append(item)
        world.event_log.append(f"take:{item}")
        return f"Tomas {obj.name}."

    return take_impl


def _make_use(world: World) -> Callable[..., str]:
    def use_impl(item: str, target: str) -> str:
        if item not in world.inventory:
            return f"Error: no llevas ningún {item!r}."
        if target not in world.items:
            return f"Error: no existe ningún objeto con id {target!r}."
        tgt = world.items[target]
        if not world.is_visible(target):
            return f"Error: no ves ningún {target!r} aquí."
        if tgt.locked is None:
            return (
                f"Usas {world.items[item].name} con {tgt.name}, pero no pasa nada."
            )
        if tgt.open_state == "open":
            return f"{tgt.name} ya está abierta."

        item_name = world.items[item].name

        # Cerradura multi-item: hay que colocar varias piezas (de distintos
        # sitios) antes de que se abra.
        required_items = tgt.locked.get("requires_items")
        if required_items:
            if item not in required_items:
                return f"Intentas usar {item_name} con {tgt.name}, pero no encaja."
            if item in tgt.inserted:
                return f"Ya habías colocado {item_name} en {tgt.name}."
            tgt.inserted.append(item)
            missing = [r for r in required_items if r not in tgt.inserted]
            if not missing:
                tgt.open_state = "open"
                world.event_log.append(f"open:{target}")
                return (
                    f"Colocas {item_name} en {tgt.name}. "
                    "Era la última pieza: se abre."
                )
            missing_names = ", ".join(
                world.items[m].name if m in world.items else m for m in missing
            )
            return (
                f"Colocas {item_name} en {tgt.name}. "
                f"Todavía faltan piezas: {missing_names}."
            )

        # Cerradura de una sola llave.
        required = tgt.locked.get("requires_item")
        if required != item:
            return f"Intentas usar {item_name} con {tgt.name}, pero no encaja."
        tgt.open_state = "open"
        world.event_log.append(f"open:{target}")
        return f"Usas {item_name} con {tgt.name}. Se abre."

    return use_impl


def _make_go(world: World) -> Callable[..., str]:
    def go_impl(direction: str) -> str:
        room = world.rooms[world.current_room]
        if not room.exits:
            return "Error: no hay salidas desde aquí."
        dest = room.exits.get(direction)
        if dest is None:
            options = ", ".join(sorted(room.exits.keys()))
            return (
                f"Error: no hay salida {direction!r} desde aquí. "
                f"Salidas disponibles: {options}."
            )
        gate_id = room.locked_exits.get(direction)
        if gate_id is not None:
            gate = world.items.get(gate_id)
            # Gate ausente => bloqueado (fail-safe): un id mal escrito no debe
            # convertir una salida cerrada en un pasillo libre.
            if gate is None or gate.open_state != "open":
                blocker = gate.name if gate is not None else gate_id
                return (
                    f"Error: el paso hacia {direction} está bloqueado por "
                    f"{blocker}. Tenés que abrirla primero."
                )
        if dest not in world.rooms:
            return (
                f"Error: la salida {direction!r} lleva a una sala "
                f"desconocida ({dest!r})."
            )
        world.current_room = dest
        world.event_log.append(f"enter:{dest}")
        return f"Caminas hacia {direction}. Llegas a {world.rooms[dest].name}."

    return go_impl


def make_world_tools(world: World) -> list[tuple[Callable[..., str], ToolSchema]]:
    """Devuelve los 4 verbos del mundo enlazados a `world`.

    Uso típico::

        for fn, schema in make_world_tools(world):
            agent.register_tool(fn, schema)

    El verbo `go` solo se incluye si el mundo tiene navegación (alguna sala
    con `exits`); en mundos de una sola sala devuelve únicamente los cuatro
    verbos clásicos.
    """
    tools: list[tuple[Callable[..., str], ToolSchema]] = [
        (_make_look(world), look_schema),
        (_make_examine(world), examine_schema),
        (_make_take(world), take_schema),
        (_make_use(world), use_schema),
    ]
    if any(room.exits for room in world.rooms.values()):
        tools.append((_make_go(world), go_schema))
    return tools
