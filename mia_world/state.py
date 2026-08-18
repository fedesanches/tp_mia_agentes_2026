"""Modelo de estado del mundo simulado (M3).

Definiciones FIJAS: la infraestructura de evaluación importa estas
dataclasses y depende de su forma. No las edites en una entrega.

Conceptos:
  - `Item`: cualquier objeto del mundo (alfombra, llave, puerta, nota...).
    Un mismo Item puede ser, a la vez, contenedor (`container`) y
    tomable (`takeable`); típicamente solo es uno de los dos.
  - `Room`: una sala con descripción y una lista de ids de Item
    directamente visibles desde ella.
  - `World`: el estado mutable. Las herramientas del agente cierran
    sobre una instancia de `World` y la modifican en sitio.
  - `Scenario`: configuración inicial inmutable + condición de victoria.

Visibilidad: un Item es visible para el agente si:
  - está directamente en `room.items` de la sala actual y no tiene
    `hidden_by`, **o**
  - su `hidden_by` está en `world.revealed` (es decir, el contenedor
    padre ya fue examinado) y sigue presente en la lista `contains`
    de ese contenedor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Item:
    id: str
    name: str
    description: str
    takeable: bool = False
    container: bool = False
    contains: list[str] = field(default_factory=list)
    hidden_by: str | None = None
    locked: dict[str, Any] | None = None
    open_state: str = "closed"  # "closed" | "open"
    # Piezas ya colocadas en una cerradura multi-item (`locked` con
    # `requires_items`). Empieza vacío; `use` lo va poblando hasta que
    # están todas y el item se abre. Irrelevante para cerraduras de una
    # sola llave (`requires_item`).
    inserted: list[str] = field(default_factory=list)


@dataclass
class Room:
    id: str
    name: str
    description: str
    items: list[str] = field(default_factory=list)
    # Salidas de navegación: dirección ("norte", "este", ...) -> id de sala
    # contigua. Vacío en escenarios de una sola sala (retrocompatible).
    exits: dict[str, str] = field(default_factory=dict)
    # Salidas con puerta: dirección -> id del Item-puerta que la bloquea.
    # `go` solo deja pasar si ese Item tiene `open_state == "open"`. Permite
    # progresión gated y backtracking. Vacío = todas las salidas libres.
    locked_exits: dict[str, str] = field(default_factory=dict)


@dataclass
class World:
    rooms: dict[str, Room]
    items: dict[str, Item]
    current_room: str
    inventory: list[str] = field(default_factory=list)
    revealed: set[str] = field(default_factory=set)
    # Bitácora ordenada de eventos semánticos que producen las
    # herramientas: "enter:<room>", "take:<item>", "open:<item>". Permite
    # comprobar goals de tipo `sequence` (orden temporal), no solo estado
    # final. Aditivo: los escenarios de una sola sala solo registran el
    # "enter:" de la sala inicial.
    event_log: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.event_log and self.current_room:
            self.event_log.append(f"enter:{self.current_room}")

    def is_visible(self, item_id: str) -> bool:
        if item_id not in self.items:
            return False
        item = self.items[item_id]
        if item.hidden_by is None:
            return item_id in self.rooms[self.current_room].items
        if item.hidden_by not in self.revealed:
            return False
        parent = self.items.get(item.hidden_by)
        # El contenedor padre debe ser visible desde la sala actual: `revealed`
        # es global, así que sin este anclaje un hijo revelado en una sala
        # seguiría siendo visible/tomable desde cualquier otra (la recursión
        # cubre además contenedores anidados).
        return (
            parent is not None
            and item_id in parent.contains
            and self.is_visible(item.hidden_by)
        )


@dataclass
class Scenario:
    id: str
    description: str
    user_message: str
    initial_world: World
    goal: dict[str, Any]
    difficulty: str = "unspecified"  # "easy" | "medium" | "hard" | "unspecified"
