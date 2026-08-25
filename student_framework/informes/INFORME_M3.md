# Informe Milestone 3 — Evaluación sobre un problema objetivo

> **Estado del documento.** Las secciones 1 y 2 (aproximación e
> infraestructura + métricas) están completas. Las secciones 3–5
> (resultados finales, experimentos y conclusiones) se completan tras
> cerrar US-01 (adaptación del agente al mundo) y correr la evaluación
> completa. Los marcadores `〔PENDIENTE …〕` señalan lo que falta llenar
> con números reales.

## 1. Aproximación

### 1.1. Cómo aplicamos el framework de M1+M2 al problema

El problema de M3 es una **sala de escape simulada** (`mia_world/`): el
agente debe manipular el mundo (abrir la puerta principal, y en algunos
escenarios recuperar objetos o navegar entre salas) usando cinco verbos
genéricos —`look`, `examine`, `take`, `use` y, en multi-sala, `go`—. El
éxito se mide sobre el **estado del mundo** con `mia_world.check_goal`, no
sobre el texto del agente, lo que da una métrica objetiva.

Reutilizamos el único punto de entrada público del framework,
`build_agent` (el mismo que usan los tests de conformidad y la CLI de
`mia_world`), sin re-implementar el bucle. Sobre la instancia devuelta,
la infraestructura registra las herramientas del mundo con
`agent.register_tool(fn, schema)`, donde los pares `(callable, ToolSchema)`
provienen de `make_world_tools(world)`. Cada herramienta cierra sobre una
instancia mutable de `World`, de modo que las acciones del agente
modifican el estado que luego evalúa `check_goal`.

Las piezas de M1+M2 que este problema ejercita directamente:

- **Bucle de tool-calling (M1):** el mundo se resuelve encadenando
  llamadas a herramientas (`look → examine → take → use`). El bucle de
  `run` y el registro por `AgentStep` son la base de la evaluación.
- **Memoria conversacional (M2):** los escenarios multi-sala
  (`apartment-keys`, `office-sequence`, `vault-combination`,
  `backtracking-vault`) exigen recordar el mapa y objetos vistos en salas
  anteriores; la ventana deslizante de M2 es lo que sostiene ese contexto
  entre pasos.
- **Robustez de herramientas (M2):** las world-tools devuelven mensajes
  de error accionables (p. ej. "no ves ningún X aquí") en vez de lanzar
  excepciones, y el agente los realimenta al historial para autocorregirse.

### 1.2. Qué especializamos

- **`max_iterations` configurable en evaluación.** El default del agente
  es 10, pero el peor caso del dataset ronda las 21 llamadas óptimas
  (`vault-combination`). La infraestructura sube el tope (default 40) por
  instancia, sin tocar `student_framework`, para no penalizar escenarios
  de horizonte largo.
- **Registro de las world-tools sobre `build_agent`.** No creamos un
  entry point paralelo: reutilizamos `build_agent` tal cual y añadimos las
  herramientas del mundo encima, replicando el patrón de `mia_world/cli.py`.

〔PENDIENTE US-01: describir aquí el prompt de sala de escape y/o el entry
point específico para M3 que adapta al agente. Ver §3 para el impacto
medido de esta adaptación.〕

## 2. Infraestructura de evaluación y métricas

### 2.1. Infraestructura reproducible (`student_framework/eval/run.py`)

Un único comando corre el agente sobre **todo** el dataset sin pasos
manuales:

```bash
python student_framework/eval/run.py            # los 8 escenarios
python student_framework/eval/run.py --scenario easy
python student_framework/eval/run.py --max-iterations 40
```

Por cada caso, el arnés construye un agente **fresco** (para que no haya
fuga de estado entre escenarios), registra las world-tools, ejecuta
`agent.run(scenario.user_message)` y verifica el goal con `check_goal`
sobre el estado final del mundo. Captura y persiste:

- **Entradas:** `user_message`, `goal`, dificultad, descripción.
- **Salidas:** respuesta final del agente (`answer`).
- **Llamadas a herramientas:** cada `AgentStep` con
  `tool_name` / `tool_input` / `tool_output` / `error`.
- **Errores:** errores del agente (`AgentResult.error`) y cualquier
  excepción de ejecución, capturada por caso (nunca propaga: si un
  escenario falla, el resto del dataset sigue corriendo).

Salidas en disco, bajo `student_framework/eval/runs/<timestamp>/`:

- `cases/<scenario_id>.json` — un registro completo por escenario.
- `summary.json` — métricas agregadas legibles por máquina (incluye
  `meta` con timestamp, proveedor LLM, `max_iterations`, versión de
  Python).
- `report.md` — resumen legible por humanos con tabla por escenario.

La reproducibilidad se apoya en: agente fresco por caso, semilla de
configuración registrada en `meta`, y verificación de goal sobre estado
(determinista dado el mundo). La única fuente de variabilidad es el LLM.

### 2.2. Métrica cuantitativa: tasa de éxito + eficiencia de llamadas

**Qué medimos.** (a) **Tasa de éxito** (`success_rate`): fracción de
escenarios en que `check_goal` da verdadero. (b) **Eficiencia de
llamadas** (`calls_over_optimal`): número de tool-calls del agente menos
el óptimo del escenario (tabla de `ENUNCIADO_M3.md`, mapeada en
`OPTIMAL_CALLS`).

**Por qué.** La tasa de éxito es la métrica natural del problema: el goal
es binario y se comprueba sobre el mundo, no sobre texto, así que no hay
ambigüedad ni "crédito parcial" inflado. La eficiencia complementa: dos
agentes pueden ambos resolver un escenario, pero uno que gasta 25 llamadas
donde el óptimo es 7 revela exploración pobre, mala memoria o pérdida de
disciplina de tool-calling. Registramos además **latencia** y **tokens**
(in/out) por su valor de coste operativo.

**Cómo se computa.** `success_rate = solved / total`; `calls_over_optimal
= num_tool_calls - optimal_calls`. Ambas se agregan en `build_summary` y
se desglosan por dificultad.

### 2.3. Métrica cualitativa

〔PENDIENTE US-03: elegir y justificar la dimensión cualitativa (rúbrica
manual o LLM-as-judge sobre las trazas). Candidata: rúbrica de "calidad de
plan" que puntúe si el agente exploró antes de actuar, si evitó acciones
redundantes y si el orden de sub-objetivos fue coherente (relevante para
`office-sequence`, que tiene goal ordenado).〕

## 3. Resultados

### 3.1. Baseline preliminar (agente sin adaptar, prompt de M1)

Como línea de base ejecutamos la infraestructura con `build_agent` **tal
cual** (system prompt de M1, orientado a calculadora/clima/lector). Este
baseline documenta el punto de partida antes de US-01:

| Escenario | Dif. | Goal | Calls | Óptimo | Errores |
|---|---|---|---|---|---|
| study-with-key | easy | ❌ | 0 | 3 | 0 |
| color-locks | medium | ❌ | 0 | 11 | 0 |
| apartment-keys | medium | ❌ | 0 | 7 | 0 |
| library-search | hard | ❌ | 0 | 7 | 0 |
| extreme-archive | extreme | ❌ | 1 | 4 | 0 |

**Lectura.** Con el prompt de M1 el agente prácticamente **no usa las
world-tools** (0–1 llamadas) y no resuelve ningún escenario: responde con
texto libre ("No puedo ayudarte…") porque su system prompt le indica que
"la mayoría de preguntas NO requieren herramientas". Esto aísla con
claridad el valor de la adaptación de US-01: sin un prompt orientado a la
tarea, el bucle y la memoria por sí solos no bastan.

〔PENDIENTE: reemplazar por la corrida completa (8 escenarios) con el
agente adaptado de US-01, incluyendo tasa de éxito global, desglose por
dificultad, latencia y tokens.〕

### 3.2. Resultados con el agente adaptado

〔PENDIENTE US-01 + corrida completa. Rellenar la tabla de 8 escenarios y
las agregadas de `summary.json`.〕

## 4. Experimentos

〔PENDIENTE US-05: al menos dos experimentos que aíslen una variable del
framework. Candidatos alineados con la infraestructura ya construida:

1. **Reducir `max_iterations`** (p. ej. 40 → 10) y medir cómo cae la tasa
   de éxito en escenarios de horizonte largo (`vault-combination`,
   `backtracking-vault`). Directamente soportado por el flag
   `--max-iterations`.
2. **Intercambiar la estrategia de prompting** (ReAct puro vs. prompt con
   planificación explícita de sub-objetivos) y comparar en el escenario de
   goal ordenado (`office-sequence`).
3. **Apagar/limitar la memoria** (`max_history_messages` chico) y medir el
   impacto en los escenarios multi-sala que exigen recordar el mapa.

Documentar, por experimento: qué se cambió, qué pasó (números), qué se
concluyó, comparado contra la configuración base de §3.〕

## 5. Limitaciones y próximos pasos

〔PENDIENTE — completar tras resultados. Puntos ya identificados:

- **Dependencia del prompt:** el baseline muestra que el comportamiento
  del agente es muy sensible al system prompt; sin adaptación no usa las
  herramientas del mundo.
- **Variabilidad del LLM:** la única fuente de no-determinismo del arnés
  es el modelo; convendría promediar varias corridas (pass@k) por
  escenario para estabilizar la métrica.
- **`OPTIMAL_CALLS` mapeado a mano:** los óptimos vienen de la tabla del
  enunciado (los JSON de escenario son FIJOS); si el dataset cambia, hay
  que actualizar el mapa.
- **Análisis de errores (US-04):** categorizar los modos de fallo
  (no exploró, se quedó sin pasos, perdió disciplina de tool-calling en
  `extreme`, error de navegación multi-sala) a partir de las trazas
  capturadas.〕
