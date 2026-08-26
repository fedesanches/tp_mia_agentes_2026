"""Implementación de su agente.

Completen `register_tool` y `run` para el Milestone 1.
En el Milestone 2 amplíen `MyAgent` para que sea estatal y respete
`max_history_messages`.

Los tests de conformidad en `tests/conformance/test_m1.py` y
`test_m2.py` describen con precisión qué comportamientos deben funcionar
— léanlos antes de empezar.
"""

from __future__ import annotations

import json
from pyexpat.errors import messages
from typing import Any, Callable
from urllib.error import URLError

from mia_agents.protocols import LLMClient
from mia_agents.types import AgentResult, AgentStep, ToolSchema
from mia_agents import final_result_tool_schema, FINAL_RESULT_TOOL_NAME
from pydantic import ValidationError, schema

# Acá definimos una lista de errores que vamos a considerar transitorios. Es decir
# que si son detectatos, el agente puede reintentar la ejecución de acción que los provoca.
_TRANSIENT_ERRORS = (TimeoutError, ConnectionError, URLError)

class MyAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = "Eres un asistente útil.",
        max_iterations: int = 10,
        max_history_messages: int = 50,
    ) -> None:
        """Inicializa el agente.

        Parameters
        ----------
        llm_client : LLMClient
            Cliente LLM (real o mock) que el agente utilizará.
        system_prompt : str
            System prompt por defecto.
        max_iterations : int
            Tope de iteraciones del bucle del agente (M1).
        max_history_messages : int
            Número máximo de mensajes que se permiten en la lista
            `messages` enviada al LLM en una única llamada. En M1 este
            valor es ignorado; el agente sólo necesita aceptarlo en su
            constructor. En M2 deben respetarlo: la longitud de la
            lista de mensajes pasada a `self._llm.chat(...)` no puede
            superar este número en ninguna llamada, sin importar la
            estrategia de memoria que elijan.
        """
        self._llm = llm_client
        self._system = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max_history_messages
        
        # Inicializa el estado interno para las herramientas registradas.
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: dict[str, ToolSchema] = {}
        
        # TODO (M2): inicializa la estructura de historial conversacional.
        self._history: list[dict[str, Any]] = []
        self._last_user_index: int = -1  # Índice del último mensaje de usuario en el historial.

    def register_tool(
        self,
        tool: Callable[..., str],
        schema: ToolSchema,
    ) -> None:
        """
        Registra una herramienta callable junto a su esquema.

        El esquema suele obtenerse con `ToolSchema.from_callable(fn)`. En
        `run`, pasá `tools=list(self._schemas.values())`; el cliente LLM
        aplica `to_llm_spec()` al llamar al proveedor.

        El callable se invoca con kwargs que coinciden con la firma.
        Debe devolver una cadena.
        """
        self._tools[schema.name] = tool
        self._schemas[schema.name] = schema

    def run(self, user_message: str) -> AgentResult:
        """
        Ejecuta el bucle del agente hasta una respuesta final o hasta max_iterations.

        Comportamiento esperado (consulta tests/conformance/test_m1.py
        para el contrato exacto del M1):
          - Llama a `self._llm.chat(..., tools=list(self._schemas.values()))`.
          - Si la respuesta contiene tool_calls, ejecuta cada uno y vuelca
            los resultados en la siguiente llamada al chat.
          - Si la respuesta solo contiene texto (sin `tool_calls`),
            devuélvelo en `AgentResult.answer`. En M1 no uses la tool
            sintética `final_result`; ese patrón es de M2 (ver README y
            ENUNCIADO_M2.md).
          - Limita el bucle a `self._max_iterations` y termina de forma
            limpia cuando se alcance.
          - Registra cada invocación de herramienta como un `AgentStep`
            dentro de `result.steps`.

        En el M2, además, llamadas sucesivas sobre la misma instancia
        deben continuar la conversación, y la longitud de la lista de
        mensajes enviada al LLM no debe superar `self._max_history_messages`.
        Acumula los tokens de entrada/salida reportados por los
        `LLMResponse` y exponlos en `AgentResult.input_tokens` /
        `AgentResult.output_tokens`.
        """
        self._history.append({"role": "user", "content": user_message})
        self._last_user_index = len(self._history) - 1
        steps: list[AgentStep] = []

        # Acumuladores de tokens para esta llamada a run().
        input_tokens: int | None = None
        output_tokens: int | None = None

        for _ in range(self._max_iterations):
            # Llamamos al LLM con la historia de mensajes y las herramientas registradas.
            # Hacemos la llamada dentro de `_call_with_retries` para manejar errores transitorios.
            resp = self._call_with_retries(
                lambda: self._llm.chat(
                    messages=self._windowed_history(),
                    tools=list(self._schemas.values()) if self._schemas else None,
                    system=self._system,
                )
            )

            if resp.input_tokens is not None:
                input_tokens = (input_tokens or 0) + resp.input_tokens
            if resp.output_tokens is not None:
                output_tokens = (output_tokens or 0) + resp.output_tokens

            # Si no hay herramientas para ejecutar, esta es la respuesta final.
            if not resp.tool_calls:
                self._history.append({"role": "assistant", "content": resp.content or ""})
                return AgentResult(
                    answer=resp.content or "",
                    steps=steps,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            # Guardamos que el asistente pidio ejecutar herramientas.
            self._history.append(
                {
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in resp.tool_calls
                    ],
                }
            )

            # Ejecutamos cada herramienta y agregamos su resultado al historial.
            for tool_call in resp.tool_calls:
                try:
                    args = json.loads(tool_call.arguments)
                    tool = self._tools[tool_call.name]

                    # Ejecutamos la herramienta con reintentos en caso de errores transitorios.
                    output = self._call_with_retries(lambda: tool(**args))
                    error = None
                except KeyError:
                    output = f"Error: herramienta desconocida '{tool_call.name}'"
                    error = output
                except json.JSONDecodeError:
                    output = f"Error: argumentos JSON inválidos para '{tool_call.name}'"
                    error = output
                except Exception as e:
                    output = f"Error: excepción al ejecutar '{tool_call.name}': {e}"
                    error = output

                steps.append(
                    AgentStep(
                        tool_name=tool_call.name,
                        tool_input=tool_call.arguments,
                        tool_output=output,
                        error=error,
                    )
                )
                self._history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    }
                )

        return AgentResult(
            answer="",
            steps=steps,
            error="Se alcanzo el limite de iteraciones sin respuesta final.",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _windowed_history(self) -> list[dict[str, Any]]:
        """Recorta self._history a cuando mucho, a max_history_messages. Sin
        descartar nunca el último mensaje de usuario que siempre debe estar presente en la ventana de 
        mensajes enviados al LLM."""
        budget = self._max_history_messages
        history = self._history

        if len(history) <= budget:
            return list(history)

        window_start = len(history) - budget
        if self._last_user_index >= window_start:
            # el mensaje de usuario ya cae dentro de la ventana de cola
            return self._drop_orphan_tool_results(list(history[window_start:]))

        # liberamos el espacio del extremo más viejo de la ventana
        last_user_message = history[self._last_user_index]
        remaining = budget - 1
        tail = history[-remaining:] if remaining > 0 else []
        tail = self._drop_orphan_tool_results(tail)

        return [last_user_message, *tail]

    @staticmethod
    def _drop_orphan_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Descarta resultados de tool al inicio de la ventana cuyo assistant
        (tool_calls) quedó fuera del recorte: sin ese toolUse previo, la API
        Converse de Bedrock rechaza el bloque toolResult huérfano."""
        start = 0
        while start < len(messages) and messages[start].get("role") == "tool":
            start += 1
        return messages[start:]

    def _call_with_retries(self, fn, max_attempts: int = 3):
        """
        Llama a `fn` hasta `max_attempts` veces si se detecta un error transitorio.
        - Si `fn` lanza un error que no está en `_TRANSIENT_ERRORS`, se propaga inmediatamente.
        - Si `fn` lanza un error transitorio, se reintenta hasta `max_attempts` veces.
        - Si tras `max_attempts` sigue fallando, se propaga la última excepción.
        """
        last_exc: Exception | None = None
        for _ in range(max_attempts):
            try:
                return fn()
            except _TRANSIENT_ERRORS as e:
                last_exc = e
                continue
        raise last_exc
        
    def structured_call(
        self,
        prompt: str,
        schema: Any,
        max_repair_attempts: int = 2,
    ) -> Any:
        """Pide al LLM una respuesta validada contra `schema` (M2).

        Obligatorio: herramienta sintética `final_result` (ver
        `mia_agents.final_result_tool_schema` / `FINAL_RESULT_TOOL_NAME`).
        El agente ofrece esa tool al LLM, valida los `arguments` del
        `tool_call` y reintenta con contexto de reparación si el modelo
        responde con texto libre o con argumentos inválidos.

        Implementa esto en el M2:
          - Pasa `tools=[final_result_tool_schema(schema)]` en cada
            llamada a `chat` dentro de este método.
          - Termina solo cuando llega un `tool_call` a `final_result`
            cuyos argumentos validan con `schema.model_validate(...)`.
          - Reintenta hasta `max_repair_attempts` incluyendo el fallo en
            los mensajes (respuesta previa, mensaje `tool`, o user de
            reparación).
          - Si tras los reintentos sigue fallando, levanta una excepción
            limpia (no devuelvas valores parciales ni `None` sin avisar).

        El M1 deja esto como stub; los tests de M2 verifican el contrato.
        """
        final_tool = final_result_tool_schema(schema)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        for attempt in range(max_repair_attempts + 1):
            # Llamamos al LLM con la historia de mensajes y la herramienta final_result.
            # Hacemos la llamada dentro de `_call_with_retries` para manejar errores transitorios.
            resp = self._call_with_retries(
                lambda: self._llm.chat(
                    messages=messages,
                    tools=[final_tool],
                    system=self._system,
                )
            )

            
            final_call = next(
                (tc for tc in resp.tool_calls if tc.name == FINAL_RESULT_TOOL_NAME), 
                None
            )

            if final_call is None:
                messages.append({"role": "assistant", "content": resp.content or ""})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Debes invocar la herramienta '{FINAL_RESULT_TOOL_NAME}' "
                        "con los argumentos requeridos. No respondas con texto libre. Intenta de nuevo."
                    ),
                })
                continue

            try:
                args = json.loads(final_call.arguments)
                return schema.model_validate(args)
            except (json.JSONDecodeError, ValidationError) as e:
                messages.append({
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [{
                        "id": final_call.id,
                        "function": {"name": final_call.name, "arguments": final_call.arguments},
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": final_call.id,
                    "content": f"Error de validación: {e}. Corregí los argumentos y volvé a invocar '{FINAL_RESULT_TOOL_NAME}'.",
            })
            continue

        raise RuntimeError(
            f"No se pudo obtener una respuesta válida de '{FINAL_RESULT_TOOL_NAME}' tras {max_repair_attempts} intentos."
        )
