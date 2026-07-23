"""
Leer archivos de texto y su esquema correspondiente.
"""
from __future__ import annotations
import os
from typing import Annotated
from pydantic import Field
from mia_agents.types import ToolSchema

# Tamaño máximo de archivo a leer (1 MB)
_MAX_BYTES = 1_000_000  # 1 MB

# Extensiones de archivo permitidas
_ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".py", ".yaml", ".yml", ".log"}


def _sandbox_root() -> str:
    """Directorio base contra el que se resuelven las rutas relativas."""
    return os.path.realpath(os.getcwd())


def _list_files(directory: str) -> list[str]:
    """Devuelve los nombres de archivos (no directorios) dentro de `directory`."""
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return []
    return [e for e in entries if os.path.isfile(os.path.join(directory, e))]


def file_reader(
    path: Annotated[str, Field(description="Ruta al archivo de texto a leer. Se admite una ruta relativa dentro del proyecto (sin '..') o una ruta absoluta a un archivo de texto existente.")],
) -> str:
    """
    Lee el contenido de un archivo de texto UTF-8.
    Devuelve un mensaje de error claro y accionable si la lectura no es
    posible, para que se puedan corregir los argumentos y reintentar.
    """
    # 1. Ruta vacía.
    if not path or not path.strip():
        return (
            "Error: la ruta esta vacia. Pasa una ruta relativa a un archivo de "
            "texto dentro del proyecto, por ejemplo 'datos/notas.txt'."
        )

    raw = path.strip()

    # 2. Navegación con '..' (traversal): no se permite en ninguna forma.
    parts = raw.replace("\\", "/").split("/")
    if ".." in parts:
        return (
            f"Error: la ruta {raw!r} contiene '..' (navegacion a directorios "
            "superiores), lo cual no esta permitido. Usa una ruta relativa sin "
            "'..', por ejemplo 'datos/notas.txt'."
        )

    root = _sandbox_root()

    # 3. Resolución de la ruta. Las relativas se anclan al sandbox; las
    #    absolutas se permiten como rutas seguras hacia archivos existentes.
    if os.path.isabs(raw):
        resolved = os.path.realpath(raw)
    else:
        resolved = os.path.realpath(os.path.join(root, raw))
        # Defensa extra ante symlinks u otras formas de escape del sandbox.
        try:
            if os.path.commonpath([root, resolved]) != root:
                return (
                    f"Error: la ruta {raw!r} escapa del directorio permitido "
                    f"({root}). Usa una ruta relativa dentro del proyecto, por "
                    "ejemplo 'datos/notas.txt'."
                )
        except ValueError:
            return (
                f"Error: la ruta {raw!r} escapa del directorio permitido "
                f"({root}). Usa una ruta relativa dentro del proyecto."
            )

    # 4. Inexistencia: si el directorio contenedor existe, listamos sus
    #    archivos para que se pueda elegir la ruta correcta.
    if not os.path.exists(resolved):
        parent = os.path.dirname(resolved) or root
        if os.path.isdir(parent):
            files = _list_files(parent)
            if files:
                return (
                    f"Error: el archivo {raw!r} no existe. En el directorio "
                    f"'{parent}' hay estos archivos disponibles: "
                    f"{', '.join(files)}."
                )
            return (
                f"Error: el archivo {raw!r} no existe y el directorio '{parent}' "
                "no contiene archivos para leer."
            )
        return (
            f"Error: la ruta {raw!r} no existe y su directorio contenedor tampoco. "
            "Verifica la ruta."
        )

    # 5. La ruta apunta a un directorio en lugar de a un archivo.
    if os.path.isdir(resolved):
        files = _list_files(resolved)
        if files:
            return (
                f"Error: la ruta {raw!r} apunta a un directorio, no a un archivo. "
                f"Archivos dentro de ese directorio: {', '.join(files)}."
            )
        return (
            f"Error: la ruta {raw!r} apunta a un directorio (sin archivos dentro), "
            "no a un archivo. Pasa la ruta a un archivo de texto."
        )

    # 6. Extensión no permitida.
    _, ext = os.path.splitext(resolved)
    if ext.lower() not in _ALLOWED_EXTENSIONS:
        permitidas = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        return (
            f"Error: la extension {ext or '(sin extension)'!r} no esta permitida. "
            f"Extensiones validas: {permitidas}."
        )

    # 7. Lectura del contenido con tope de tamaño.
    try:
        with open(resolved, encoding="utf-8") as f:
            content = f.read(_MAX_BYTES)
            if f.read(1):
                return (
                    f"Error: el archivo {raw!r} supera el tamaño maximo permitido "
                    f"de {_MAX_BYTES} bytes."
                )
            return content
    except UnicodeDecodeError:
        return (
            f"Error: el archivo {raw!r} no es texto UTF-8 valido; solo se pueden "
            "leer archivos de texto."
        )
    except OSError as e:
        return f"Error al leer el archivo {raw!r}: {e}"


file_reader_schema = ToolSchema.from_callable(file_reader)
