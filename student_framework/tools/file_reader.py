"""
Leer archivos de texto y su esquema correspondiente.
"""
from __future__ import annotations
from typing import Annotated
from pydantic import Field
from mia_agents.types import ToolSchema

# Tamaño máximo de archivo a leer (1 MB)
_MAX_BYTES = 1_000_000  # 1 MB

# Extensiones de archivo permitidas
_ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".py", ".yaml", ".yml", ".log"}

def file_reader(
    path: Annotated[str, Field(description="Ruta al archivo de texto a leer.")],
) -> str:
    """
    Lee el contenido de un archivo de texto UTF-8 dentro del workspace.
    Devuelve un mensaje de error legible si la lectura no es posible.
    """
    if not any(path.endswith(ext) for ext in _ALLOWED_EXTENSIONS):
        return f"Error: La extensión del archivo no está permitida. Extensiones permitidas: {_ALLOWED_EXTENSIONS}"

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read(_MAX_BYTES)
            if f.read(1):
                return f"Error: El archivo es demasiado grande. Tamaño máximo permitido: {_MAX_BYTES} bytes."
            return content
    except Exception as e:
        return f"Error al leer el archivo: {e}"


file_reader_schema = ToolSchema.from_callable(file_reader)
