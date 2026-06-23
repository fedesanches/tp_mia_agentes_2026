"""
Leer archivos de texto y su esquema correspondiente.
"""
from __future__ import annotations
from typing import Annotated
from pydantic import Field
from mia_agents.types import ToolSchema

def file_reader(
    path: Annotated[str, Field(description="Ruta al archivo de texto a leer.")],
) -> str:
    """
    Lee el contenido de un archivo de texto y lo devuelve como cadena.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


file_reader_schema = ToolSchema.from_callable(file_reader)
