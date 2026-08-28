# Pipeline de ejecución del agente (M1 + capacidades nuevas de M2)

Explicación funcional de base, previa a abordar M3. Reúne lo que importa
de M1 y M2 para construir sobre ello: el ciclo genérico de herramientas
de M1, y las dos capacidades nuevas que agregó M2 (memoria y salida
estructurada). Lo que M2 solo *robustece* (sliding window, manejo de
errores en tools, reintentos ante fallos transitorios) queda fuera de
este pipeline — sirve como referencia, no es parte del flujo conceptual.

## Ciclo del agente (núcleo M1, unificado con M3)

El ciclo genérico de tool-use no cambia entre M1 y M3: solo se amplía
horizontalmente el set de tools disponible en el paso 3, **sobre la
misma instancia** que ya arma `build_agent()` (confirmado en
`mia_world/cli.py`: registra las tools del mundo encima de las de M1, no
en una instancia aparte). Lo único que agrega M3 es una verificación
externa, después de una única llamada a `run`, que reemplaza "el LLM
dice que terminó" por "el estado del mundo dice que terminó":

```
build_agent() → ya registradas: calculator, file_reader, current_temperature
        +
register_tool: look, examine, take, use, go     (misma instancia)
        ↓
   ┌────────────────────────────────────────────────────────────┐
   │ 1. Llega la instrucción del escenario                       │
   │        ↓                                                    │
   │ 2. El agente le manda al LLM: instrucción + system prompt   │
   │    + schemas de TODAS las tools registradas                 │
   │        ↓                                                    │
   │ 3. El LLM decide:                                           │
   │    ├── TEXTO LIBRE     →  fin del ciclo                     │
   │    └── invoca una TOOL →  el agente ejecuta el callable      │
   │             ↓                                               │
   │        resultado se agrega como contexto                    │
   │             ↓                                               │
   │        vuelve al paso 2                                     │
   │        (corta también por límite de pasos, max_iterations)  │
   └────────────────────────────────────────────────────────────┘
        ↓
[check_goal(world)]  →  única verificación, SIN reintento de turno
```

Este ciclo es **agnóstico de la herramienta**: el agente no sabe si está
calculando, leyendo un archivo, o mirando una sala de escape. Solo sabe
"tengo tools registradas, le pregunto al LLM, ejecuto lo que pida".

Quién hace qué: el **LLM** interpreta la intención y decide qué tool
invocar; el **agente** (código Python) es un ejecutor mecánico. El
`system_prompt` importa acá: si no menciona genéricamente las tools de
exploración, el LLM puede responder con texto libre en el paso 3 sin
invocar ninguna (confirmado empíricamente con `study-with-key`).

## Capacidad nueva 1 (M2) — Memoria

No es un paso nuevo dentro del ciclo — es un cambio en **qué tan largo es
el ciclo y qué recuerda entre ejecuciones**. En M1, el paso 1 (mensaje de
usuario) reiniciaba todo desde cero en cada llamada. En M2, ese mismo
ciclo de arriba se repite sobre un historial que **persiste y se acumula
entre llamadas**, no una lista que se descarta. El pipeline es el mismo;
deja de ser "una foto" y pasa a ser "una película continua".

## Capacidad nueva 2 (M2) — Salida estructurada

Es una **variante del paso 3**, no un agregado al medio del ciclo. En vez
de que el ciclo termine apenas el LLM da texto libre, se le exige invocar
una tool especial (`final_result`) con datos validables contra un schema.
Si falla la validación, o si responde con texto libre en vez de invocar
esa tool, no termina — se le informa el error como contexto y se repite
el ciclo (una reparación), hasta lograrlo o agotar los intentos.

```
3'. El LLM debe invocar `final_result(datos)`
        ├── datos válidos según el schema  →  fin, se devuelve el objeto validado
        └── datos inválidos / texto libre  →  se informa el error, se repite el ciclo
```

## Qué queda fuera de este pipeline (solo como referencia)

- Sliding window (protege la memoria en conversaciones largas, no la crea).
- Manejo de errores recuperables en `calculator`/`file_reader`.
- Reintentos ante fallos transitorios de infraestructura.

Ninguna de estas agrega un paso nuevo al flujo conceptual — hacen más
robusto lo que ya está descripto arriba.

Nota: confirmado en `mia_world/cli.py` — M3 extiende la misma instancia
que ya arma `build_agent()` (no crea una instancia nueva y aislada). El
diagrama de arriba ya lo refleja.
