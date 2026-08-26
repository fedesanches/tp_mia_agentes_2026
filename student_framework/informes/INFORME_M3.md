# Informe Milestone 3 — Evaluación sobre un problema objetivo

> **Estado del documento.** Completo: aproximación (§1), infraestructura y
> métricas (§2), resultados con el agente adaptado (§3), experimentos (§4)
> y limitaciones (§5). Los números provienen de corridas reales del arnés
> sobre Bedrock `nova-lite`; ver la fecha de cada corrida en los reportes
> bajo `student_framework/eval/runs/`.

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

### 1.3. Adaptación del agente a la sala de escape (US-01)

La adaptación vive en el **system prompt** de `build_agent`: sobre el
prompt de M1 (calculadora/clima/lector) se añade una sección de estrategia
que se activa cuando el agente detecta world-tools (mirar/examinar/tomar/
usar/moverse). Las reglas clave, derivadas de los modos de fallo
observados en las trazas: (1) mirar al entrar a una sala y usar **solo**
ids reportados (no inventar); (2) seguir **cadenas de llaves**
re-examinando un contenedor tras abrirlo para tomar lo de adentro;
(3) **cerraduras multi-pieza**: juntar todas las piezas y usarlas sobre el
objetivo; (4) respetar el **orden** cuando la tarea lo exige (acciones
irreversibles); (5) no repetir acciones ya hechas o fallidas; (6) cortar
el bucle y responder con texto final al cumplir el objetivo. El impacto
medido de esta adaptación está en §3.2 y §4.2.

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

### 2.3. Métrica cualitativa: rúbrica de calidad de proceso (determinista)

**Qué medimos.** Una **rúbrica programática** que puntúa la *calidad del
proceso* de resolución a partir de la traza de `steps` (no del texto libre
del agente), en cuatro dimensiones de 0–2 puntos cada una (total 0–8):

| Dimensión | Qué premia | Cómo se computa sobre la traza |
|---|---|---|
| **Exploración antes de actuar** | mirar/inspeccionar antes de manipular | 2 si hubo `look`/`examine` antes de la primera acción (`use`/`take`); 1 si exploró pero no antes; 0 si nunca exploró. |
| **Sin acciones redundantes** | no repetir trabajo | 2 si no hay llamadas `(tool, args)` idénticas repetidas; 1 si ≤25 %; 0 si más. |
| **Sin alucinar ids** | usar objetos que existen | 2 si ningún `tool_output` indica id inexistente/no visible; 1 si ≤25 %; 0 si más. |
| **Recuperación de errores** | reaccionar a un fallo cambiando de acción | 2 si tras ≥80 % de los pasos con error cambió de llamada; 1 si ≥40 %; 0 si insistió (o no había errores → 2, nada que recuperar). |

El puntaje por caso se guarda en `case["rubric"]` (con `total`, `max` y
`normalized`) y el promedio por dimensión en `summary["rubric_avg"]`; el
`report.md` incluye una tabla "Calidad de proceso" con una fila por
escenario y una fila de promedio.

**Por qué la elegimos (y por qué esta y no LLM-as-judge).** La métrica
cuantitativa (éxito) es **binaria** y no explica *por qué* un agente es
mejor que otro: dos agentes con la misma tasa de éxito pueden diferir en si
exploraron con método o avanzaron por prueba y error. La rúbrica captura
esa dimensión de calidad. Entre las dos formas que admite el enunciado
—rúbrica vs. LLM-as-judge— elegimos la **rúbrica determinista** porque:

- **Reproducibilidad**: es el objetivo central de US-02. La rúbrica es una
  función pura de la traza: dos corridas con la misma traza dan el mismo
  puntaje, sin la variabilidad de un juez LLM.
- **Costo y velocidad**: no consume tokens ni añade latencia; puede correr
  sobre todos los casos y en cada experimento de US-05 sin coste marginal.
- **Auditabilidad**: cada punto se deriva de una regla explícita sobre
  `steps`, así que un fallo de puntaje es inspeccionable, no una opinión.

**Contrapartida asumida.** Mide comportamiento *observable* en la traza,
no la calidad del razonamiento interno (que sí capturaría un LLM-as-judge).
Lo aceptamos a favor de reproducibilidad y costo; un juez LLM queda
propuesto como trabajo futuro (§5).

**Cómo se computa.** Implementada en `score_rubric(steps)` en
`student_framework/eval/run.py`; se invoca por caso en `run_scenario` y se
agrega en `build_summary`.

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

### 3.2. Resultados con el agente adaptado

Corrida completa de los 8 escenarios con el agente adaptado (US-01),
proveedor **Bedrock `amazon.nova-lite-v1:0`**, `max_iterations=40`:

| Escenario | Dif. | Goal | Calls | Óptimo | Errores | Latencia (s) |
|---|---|---|---|---|---|---|
| study-with-key | easy | ✅ | 6 | 3 | 0 | 5.7 |
| color-locks | medium | ✅ | 15 | 11 | 0 | 14.1 |
| library-search | hard | ✅ | 17 | 7 | 0 | 15.1 |
| extreme-archive | extreme | ✅ | 23 | 4 | 0 | 26.4 |
| apartment-keys | medium | ✅ | 12 | 7 | 0 | 10.0 |
| office-sequence | hard | ✅ | 21 | 13 | 0 | 17.6 |
| vault-combination | extreme | ❌ | 40 | 21 | 0 | 33.1 |
| backtracking-vault | extreme | ❌ | 40 | 18 | 0 | 39.5 |

**Agregados** (`summary.json`): **6/8 resueltos** (éxito 0.75) · latencia
media 20.2 s · 816 121 tok in / 10 378 out · **0 casos con error de
ejecución**. Por dificultad: easy 1/1, medium 2/2, hard 2/2, extreme 1/3.

**Rúbrica de calidad de proceso (US-03, 0–8).** Promedio **6.25/8**
(normalizado 0.781):

| Escenario | Explor. | No-red. | No-halu. | Recup. | Total |
|---|---|---|---|---|---|
| study-with-key | 2 | 0 | 2 | 2 | 6/8 |
| color-locks | 2 | 1 | 2 | 2 | 7/8 |
| library-search | 2 | 1 | 2 | 2 | 7/8 |
| extreme-archive | 2 | 2 | 2 | 2 | 8/8 |
| apartment-keys | 2 | 0 | 1 | 2 | 5/8 |
| office-sequence | 2 | 0 | 2 | 2 | 6/8 |
| vault-combination | 2 | 0 | 1 | 2 | 5/8 |
| backtracking-vault | 2 | 0 | 2 | 2 | 6/8 |
| **promedio** | 2.0 | 0.5 | 1.75 | 2.0 | **6.25** |

**Lectura.** El agente **siempre explora antes de actuar** (Explor. 2.0) y
**se recupera de los errores** (Recup. 2.0); el punto flojo es la
**redundancia** (No-red. 0.5): incluso cuando resuelve, repite acciones
(p. ej. re-examinar objetos ya vistos), lo que explica que gaste más
llamadas que el óptimo. Los dos `extreme` no resueltos igual puntúan 5–6/8
en la rúbrica: fallan por **no completar el objetivo** (planificación de
largo horizonte), no por explorar mal.

**Bug destapado por el arnés.** La primera corrida completa con Bedrock
falló en 5/8 escenarios con `ValidationException` de la API Converse
("toolResult blocks exceeds toolUse blocks"). El eval capturó el traceback
por caso y permitió localizar la causa en `MyAgent._windowed_history`:
al recortar el historial a `max_history_messages` podía cortar en medio de
un grupo `assistant(tool_calls)`+resultados, dejando bloques `toolResult`
huérfanos que Bedrock rechaza. Se corrigió con
`MyAgent._drop_orphan_tool_results`, que descarta esos resultados sin su
`toolUse` previo. Esto ilustra el valor de la infraestructura de US-02: el
arnés reproducible convirtió un fallo intermitente en un caso localizable.

## 4. Experimentos

Cada experimento aísla **una** variable y compara contra la configuración
base de §3.2 (agente adaptado, Bedrock nova-lite, `max_iterations=40`).

### 4.1. Impacto del recorte de historial (fix de windowing)

**Qué se cambió.** Única variable: la presencia del fix
`_drop_orphan_tool_results` en `_windowed_history` (todo lo demás igual,
mismo prompt).

| Config | Resueltos | Casos con error de ejecución |
|---|---|---|
| Sin fix | 3/8 | 5 (`ValidationException` Converse) |
| Con fix | 5/8 | 0 |

**Conclusión.** El recorte de memoria por sí solo podía **romper** el
contrato de mensajes de Bedrock en los escenarios largos (los que superan
`max_history_messages` y por ende disparan el recorte). El fix elimina el
100 % de los crashes y sube la tasa de éxito 3/8 → 5/8 sin tocar la
estrategia del agente. Es un caso donde una decisión de memoria (US-02/M2)
interactúa con el proveedor concreto (US-01).

### 4.2. Guiado de estrategia en el system prompt (rendimientos decrecientes)

**Qué se cambió.** Única variable: el detalle del system prompt de
`build_agent`, en tres niveles — (a) *básico*: solo la regla "explorá
antes de actuar, no inventes ids"; (b) *v1*: seis reglas de estrategia
(cadenas de llaves, re-examinar contenedores, cerraduras multi-pieza,
orden, anti-repetición, cortar el bucle); (c) *v2*: v1 + dos reglas extra
apuntadas a los `extreme` (salida bloqueada → abrir puerta; volver a la
sala del panel remoto).

| Prompt | Resueltos | color-locks | office-seq | vault-comb | backtrack |
|---|---|---|---|---|---|
| básico | 5/8 | ❌ | ❌ | ❌ | ✅ |
| v1 (6 reglas) | **6/8** | ✅ | ✅ | ❌ | ❌ |
| v2 (8 reglas) | 5/8 | ❌ | ✅ | ❌ | ❌ |

**Conclusión.** Pasar de *básico* a *v1* resuelve dos escenarios de
razonamiento medio (color-locks, office-sequence) que pedían seguir una
cadena de contenedores y respetar un orden. Pero *v2*, con reglas más
específicas, **no** resuelve los dos `extreme` y además **desestabiliza**
color-locks. Con un modelo pequeño (nova-lite) el prompt tiene
**rendimientos decrecientes**: más instrucciones diluyen la atención y no
compensan la falta de planificación de largo horizonte. La comparación
también expone la **alta varianza por corrida** (ver §5): backtracking-vault
pasa con *básico* y falla con *v1/v2*, lo que sugiere que parte de estas
diferencias es ruido, no señal. Se adoptó *v1* como configuración base.

### 4.3. Experimentos propuestos (no ejecutados)

- **`max_iterations` (40 → 10):** medir la caída de éxito en horizonte
  largo; soportado por el flag `--max-iterations`.
- **Memoria acotada (`max_history_messages` chico):** impacto en escenarios
  multi-sala que exigen recordar el mapa.

## 5. Limitaciones y próximos pasos

- **Varianza del LLM (limitación principal de la métrica).** La única
  fuente de no-determinismo del arnés es el modelo (temperatura 0.2, sin
  seed). Observamos que un mismo prompt da 5/8 o 6/8 y que escenarios
  individuales cambian de resultado entre corridas. **El pass/fail de una
  sola corrida no es confiable.** Próximo paso concreto: agregar un flag
  `--repeat N` al harness y reportar **tasa de éxito media ± desvío**
  (pass@k) por escenario.
- **Techo del modelo en horizonte largo.** vault-combination y
  backtracking-vault (`extreme`) exigen planificación de varios pasos y
  backtracking (volver al vestíbulo a colocar 3 núcleos; abrir una puerta
  bloqueada con la llave correcta en vez de reintentar el movimiento).
  nova-lite los resuelve de forma inconsistente; no es falta de
  instrucción sino de capacidad de planificación. Un modelo más grande o
  un patrón de planificación explícita (plan-then-act) es el siguiente
  paso.
- **Redundancia de acciones.** Aun cuando resuelve, el agente repite
  acciones (No-red. 0.5/2 en la rúbrica) y gasta más llamadas que el
  óptimo. Es el margen de mejora más claro de la estrategia.
- **Dependencia del prompt.** El baseline (§3.1) muestra que sin
  adaptación el agente no usa las world-tools; el comportamiento es muy
  sensible al system prompt, con rendimientos decrecientes (§4.2).
- **`OPTIMAL_CALLS` mapeado a mano.** Los óptimos vienen de la tabla del
  enunciado (los JSON de escenario son FIJOS); si el dataset cambia, hay
  que actualizar el mapa.
- **Análisis de errores (US-04).** Las trazas capturadas permiten
  categorizar los modos de fallo ya identificados: no completar
  cerraduras multi-pieza, perder el objetivo tras juntar las piezas,
  loops de navegación, y (antes del fix) el crash de windowing.
