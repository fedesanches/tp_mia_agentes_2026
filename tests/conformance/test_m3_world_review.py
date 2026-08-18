"""Regresión de la review en profundidad de `mia_world`.

Cada bloque cubre un bug encontrado en la review y su fix:

  - BUG 1: `is_visible` filtraba items ocultos entre salas.
  - BUG 2: `check_goal` con `item_open` lanzaba KeyError ante un id inexistente.
  - BUG 3: `examine` reventaba si `contains` referenciaba un id inexistente.
  - BUG 5: una salida con puerta (`locked_exits`) hacia un item inexistente
    quedaba transitable.
  - Raíz: `load_scenario` no validaba integridad referencial; ahora falla
    en carga con un `ValueError` descriptivo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mia_world import check_goal, load_scenario, make_world_tools
from mia_world.state import Item, Room, World


def _tools(world: World) -> dict:
    return {schema.name: fn for fn, schema in make_world_tools(world)}


# ---------------------------------------------------------------------------
# BUG 1 — visibilidad de items ocultos acotada a la sala actual.
# ---------------------------------------------------------------------------


def _two_room_hidden_world() -> World:
    """Sala A con una caja (contenedor) que oculta una llave tomable; sala B vacía."""
    items = {
        "caja": Item(
            id="caja",
            name="caja",
            description="Una caja.",
            container=True,
            contains=["llave"],
        ),
        "llave": Item(
            id="llave",
            name="llave",
            description="Una llave.",
            takeable=True,
            hidden_by="caja",
        ),
    }
    rooms = {
        "A": Room(id="A", name="Sala A", description="...", items=["caja"], exits={"norte": "B"}),
        "B": Room(id="B", name="Sala B", description="...", exits={"sur": "A"}),
    }
    return World(rooms=rooms, items=items, current_room="A")


def test_hidden_child_visible_in_its_own_room() -> None:
    """Sin regresión: en la sala del contenedor, tras examinar se ve y se toma."""
    world = _two_room_hidden_world()
    tools = _tools(world)
    tools["examine"](target="caja")
    assert "llave" in tools["look"]().lower()
    assert "Tomas" in tools["take"](item="llave")
    assert "llave" in world.inventory


def test_look_hides_revealed_child_from_other_room() -> None:
    world = _two_room_hidden_world()
    tools = _tools(world)
    tools["examine"](target="caja")
    tools["go"](direction="norte")
    assert "llave" not in tools["look"]().lower()


def test_hidden_item_not_takeable_from_other_room() -> None:
    """Un item oculto en la sala A no puede tomarse estando en la sala B."""
    world = _two_room_hidden_world()
    tools = _tools(world)
    tools["examine"](target="caja")
    tools["go"](direction="norte")

    msg = tools["take"](item="llave")
    assert "Error" in msg
    assert "llave" not in world.inventory


def test_hidden_item_not_examinable_from_other_room() -> None:
    world = _two_room_hidden_world()
    tools = _tools(world)
    tools["examine"](target="caja")
    tools["go"](direction="norte")

    assert "Error" in tools["examine"](target="llave")


def test_carried_item_examinable_from_any_room() -> None:
    """Sin regresión: lo que llevás en el inventario sigue examinable en otra sala."""
    world = _two_room_hidden_world()
    tools = _tools(world)
    tools["examine"](target="caja")
    tools["take"](item="llave")
    tools["go"](direction="norte")
    assert "Error" not in tools["examine"](target="llave")


# ---------------------------------------------------------------------------
# BUG 2 — check_goal item_open protege el id como sus hojas hermanas.
# ---------------------------------------------------------------------------


def _empty_world() -> World:
    return World(rooms={"A": Room(id="A", name="Sala A", description="...")}, items={}, current_room="A")


@pytest.mark.parametrize(
    "goal",
    [
        {"type": "item_open", "item": "fantasma"},
        {"type": "agent_in_room", "room": "fantasma"},
        {"type": "item_in_inventory", "item": "fantasma"},
    ],
)
def test_check_goal_leaves_graceful_on_unknown_id(goal: dict) -> None:
    """Ninguna hoja lanza ante un id inexistente; todas devuelven (False, motivo)."""
    ok, reason = check_goal(_empty_world(), goal)
    assert ok is False
    assert isinstance(reason, str) and reason


# ---------------------------------------------------------------------------
# BUG 3 — examine tolera hijos inexistentes (igual que look).
# ---------------------------------------------------------------------------


def test_examine_container_with_dangling_child_returns_string() -> None:
    items = {
        "caja": Item(id="caja", name="caja", description="...", container=True, contains=["fantasma"]),
    }
    rooms = {"A": Room(id="A", name="Sala A", description="...", items=["caja"])}
    world = World(rooms=rooms, items=items, current_room="A")

    msg = _tools(world)["examine"](target="caja")
    assert isinstance(msg, str)
    assert "fantasma" not in msg


# ---------------------------------------------------------------------------
# BUG 5 — salida con puerta inexistente queda bloqueada (fail-safe).
# ---------------------------------------------------------------------------


def test_locked_exit_with_unknown_gate_item_stays_blocked() -> None:
    rooms = {
        "A": Room(
            id="A",
            name="Sala A",
            description="...",
            exits={"norte": "B"},
            locked_exits={"norte": "gate_inexistente"},
        ),
        "B": Room(id="B", name="Sala B", description="...", exits={"sur": "A"}),
    }
    world = World(rooms=rooms, items={}, current_room="A")

    msg = _tools(world)["go"](direction="norte")
    assert "Error" in msg and "bloqueado" in msg.lower()
    assert world.current_room == "A"


# ---------------------------------------------------------------------------
# Raíz — load_scenario valida integridad referencial y falla en carga.
# ---------------------------------------------------------------------------


def _minimal_scenario_dict() -> dict:
    return {
        "id": "review-test",
        "difficulty": "easy",
        "description": "escenario mínimo válido para tests de validación",
        "user_message": "salí",
        "start_room": "sala",
        "rooms": {"sala": {"name": "Sala", "description": "...", "items": ["puerta"]}},
        "items": {
            "puerta": {
                "name": "puerta",
                "description": "...",
                "locked": {"requires_item": "llave"},
                "open_state": "closed",
            },
            "llave": {"name": "llave", "description": "...", "takeable": True},
        },
        "goal": {"type": "item_open", "item": "puerta"},
    }


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "scenario.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_scenario_accepts_valid_world(tmp_path: Path) -> None:
    scenario = load_scenario(_write(tmp_path, _minimal_scenario_dict()))
    assert scenario.id == "review-test"


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda d: d["items"]["puerta"].__setitem__("locked", {"requires_item": "fantasma"}), "requires_item"),
        (lambda d: d["rooms"]["sala"]["items"].append("fantasma"), "items"),
        (lambda d: d["items"].__setitem__("caja", {"name": "c", "description": ".", "container": True, "contains": ["fantasma"]}), "contains"),
        (lambda d: d["rooms"]["sala"].__setitem__("exits", {"norte": "sala_fantasma"}), "exits"),
        (lambda d: (d["rooms"]["sala"].__setitem__("exits", {"norte": "sala"}), d["rooms"]["sala"].__setitem__("locked_exits", {"norte": "gate_fantasma"})), "locked_exits"),
        (lambda d: d.__setitem__("goal", {"type": "item_open", "item": "fantasma"}), "goal"),
    ],
)
def test_load_scenario_rejects_dangling_reference(tmp_path: Path, mutate, needle: str) -> None:
    data = _minimal_scenario_dict()
    mutate(data)
    with pytest.raises(ValueError) as exc:
        load_scenario(_write(tmp_path, data))
    assert needle in str(exc.value)
