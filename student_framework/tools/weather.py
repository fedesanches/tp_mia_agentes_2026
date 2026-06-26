"""Herramienta de clima actual usando Open-Meteo."""

from __future__ import annotations

import json
from typing import Annotated, Any
from urllib.parse import urlencode
from urllib.request import urlopen

from pydantic import Field

from mia_agents.types import ToolSchema


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def current_temperature(
    city: Annotated[str, Field(description="Nombre de la ciudad.")],
) -> str:
    """Devuelve la temperatura actual en grados Celsius para una ciudad."""
    geocoding_params = urlencode(
        {
            "name": city,
            "count": 1,
            "language": "es",
            "format": "json",
        }
    )
    geocoding_url = f"https://geocoding-api.open-meteo.com/v1/search?{geocoding_params}"
    geocoding_data = _get_json(geocoding_url)
    results = geocoding_data.get("results") or []

    if not results:
        return f"Error: no encontre la ciudad {city}."

    location = results[0]
    latitude = location["latitude"]
    longitude = location["longitude"]
    city_name = location.get("name", city)
    country = location.get("country", "")

    weather_params = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m",
            "temperature_unit": "celsius",
        }
    )
    weather_url = f"https://api.open-meteo.com/v1/forecast?{weather_params}"
    weather_data = _get_json(weather_url)
    temperature = weather_data["current"]["temperature_2m"]
    unit = weather_data["current_units"]["temperature_2m"]

    place = city_name if not country else f"{city_name}, {country}"
    return f"La temperatura actual en {place} es {temperature} {unit}."


current_temperature_schema = ToolSchema.from_callable(current_temperature)
