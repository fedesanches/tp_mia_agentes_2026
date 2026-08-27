# Informe Milestone 3 — Evaluación sobre un problema objetivo

> **Estado del documento.** Completo: aproximación (§1), infraestructura y
> métricas (§2), resultados y análisis de errores (§3), experimentos (§4)
> y limitaciones (§5). Los números provienen de corridas reales del arnés
> sobre Bedrock `nova-lite`; ver la fecha de cada corrida en los reportes
> bajo `student_framework/eval/runs/`.
>
> Los resultados principales (§3.2 y §3.3) corresponden a la **corrida
> canónica** `runs/20260827T221119Z`: los 8 escenarios × 3 repeticiones,
> 24 ejecuciones. Los experimentos de §4 se corrieron antes, con la misma
> configuración base pero sin repeticiones salvo donde se indica.

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
- **Repeticiones para promediar la varianza (`--repeat N`).** El arnés
  puede correr cada escenario N veces y reportar la tasa de éxito media
  (pass@k) con una tabla de estabilidad por escenario. Como
  `Scenario.initial_world` es un objeto `World` mutable **compartido**,
  cada corrida parte de una `copy.deepcopy` fresca para no arrastrar
  estado entre repeticiones. Se usa en el experimento de §4.3.

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
python student_framework/eval/run.py --repeat 3  # promediar N corridas
python student_framework/eval/run.py --reclassify runs/<timestamp>
```

El último no ejecuta al agente: reprocesa las trazas ya guardadas de una
corrida y reescribe sus artefactos. Sirve para ajustar la taxonomía de
modos de fallo (§3.3) y ver el efecto sobre datos reales en segundos, sin
volver a gastar tokens.

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
  `meta` con timestamp, proveedor LLM, `max_iterations`, `repeat`, versión
  de Python, y `by_scenario` con tasa/calls/rúbrica medios cuando N>1).
- `report.md` — resumen legible por humanos con tabla por escenario (o
  tabla de **estabilidad por escenario** cuando `--repeat` > 1).

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

**Corrida canónica** (`runs/20260827T221119Z`): los 8 escenarios con el
agente adaptado (US-01), proveedor **Bedrock `amazon.nova-lite-v1:0`**,
`max_iterations=40` y **3 repeticiones por escenario** — 24 corridas.

Reportamos sobre repeticiones y no sobre una corrida única por una razón
que los propios datos justifican: con una sola pasada, `color-locks` y
`library-search` cambian de resultado entre ejecuciones idénticas. Un
pass/fail aislado en esos escenarios describe una tirada, no al agente
(ver §5).

| Escenario | Dif. | Éxito (k/N) | Tasa | Calls medio | Óptimo | Rúbrica media |
|---|---|---|---|---|---|---|
| study-with-key | easy | 3/3 | 1.00 | 6.3 | 3 | 6.7 |
| color-locks | medium | 1/3 | 0.33 | 24.7 | 11 | 4.3 |
| apartment-keys | medium | 3/3 | 1.00 | 13.7 | 7 | 5.7 |
| library-search | hard | 2/3 | 0.67 | 16.3 | 7 | 7.7 |
| office-sequence | hard | 2/3 | 0.67 | 35.3 | 13 | 5.7 |
| extreme-archive | extreme | 3/3 | 1.00 | 23.3 | 4 | 7.7 |
| vault-combination | extreme | 0/3 | 0.00 | 46.7 | 21 | 5.3 |
| backtracking-vault | extreme | 1/3 | 0.33 | 34.3 | 18 | 5.0 |

**Agregados** (`summary.json`): **15/24 corridas exitosas** (tasa media
**0.625**) · **0 casos con error de ejecución** · 2 681 136 tok in /
34 099 out. Por dificultad: easy 3/3, medium 4/6, hard 4/6, extreme 4/9.

**Latencia: mediana 24.8 s** (media 40.9 s). Reportamos la mediana porque
una única corrida de `library-search` tardó **381.6 s** frente a 13–14 s
de sus dos repeticiones, con *menos* llamadas que ellas: es un pico del
proveedor, no del agente. Sin ese caso la media cae a 26.1 s. Es un
ejemplo de por qué la media sola engaña con n chico.

**Rúbrica de calidad de proceso (US-03, 0–8).** Promedio **6.0/8**
(normalizado 0.75), sobre las 24 corridas:

| Dimensión | Promedio |
|---|---|
| Exploración antes de actuar | 1.92 |
| Sin acciones redundantes | **0.54** |
| Sin alucinar ids | 1.63 |
| Recuperación de errores | 1.92 |
| **Total** | **6.0** |

**Lectura.** Tres resultados que la tasa agregada no muestra:

1. **La dificultad nominal no predice el éxito.** `extreme-archive`
   —diseñado para *no caber* en la ventana de contexto— sale **3/3**, con
   23 llamadas medias y rúbrica 7.7. En cambio `color-locks` (medium) sale
   1/3. Lo que separa a los escenarios no es su etiqueta sino si exigen
   **planificación de largo horizonte con vuelta atrás**: los tres peores
   (`vault-combination`, `backtracking-vault`, `color-locks`) son cadenas
   largas de dependencias, no problemas de contexto.
2. **La redundancia es el punto flojo estructural** (No-red. 0.54/2), y se
   confirma en la eficiencia: el agente gasta entre 2× y 6× las llamadas
   óptimas incluso cuando resuelve.
3. **`vault-combination` es el único 0/3.** Cuando un escenario falla
   siempre, ya no es varianza del modelo sino un techo real (§3.3).

**Nota sobre el conteo de llamadas.** `vault-combination` registra 46 y 54
llamadas con `max_iterations=40`. No es un desborde del tope: el bucle
cuenta *iteraciones*, y nova-lite a veces pide **varias herramientas en un
mismo turno**, cada una registrada como un `AgentStep`. El tope se respeta;
lo que no es 1:1 es la equivalencia iteración↔llamada.

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

### 3.3. Análisis de errores: modos de fallo (US-04)

Una tasa de éxito de 0.625 dice *cuánto* falla el agente, no *de qué*. Y el
número agregado invita a leer los fallos como si fueran el mismo fallo
repetido, cuando no lo son. Esta sección los categoriza.

#### Cómo se computa

Una **taxonomía determinista** sobre la traza de `steps`, en la misma
línea de diseño que la rúbrica de §2.3: sin coste de LLM, reproducible y
auditable. Implementada en `classify_failure` (`student_framework/eval/run.py`);
el desglose se agrega en `build_summary` y se imprime en cada `report.md`.

Las reglas se evalúan **en orden, de específico a genérico**, y gana la
primera que dispara:

| # | Categoría | Señal |
|---|---|---|
| 1 | `infra_error` | la corrida falló antes de evaluar al agente |
| 2 | `terminacion_prematura` | el bucle cerró solo, sin agotar el presupuesto |
| 3 | `id_alucinado` | ≥20 % de pasos sobre ids inexistentes o no visibles |
| 4 | `loop_navegacion` | ≥25 % de pasos `go` que devolvieron error |
| 5 | `loop_improductivo` | ≥10 % de pasos sobre un estado ya alcanzado |
| 6 | `loop_estancado` | ≥55 % de llamadas repetidas hasta agotar el presupuesto |
| 7 | `objetivo_incompleto` | (fallback) |

El orden importa y no es cosmético: `vault-combination` presenta a la vez
76 % de llamadas repetidas y 26 % de errores de navegación. Clasificar por
frecuencia dominante lo etiquetaría como repetición, ocultando que el
agente estuvo chocando contra una pared. La repetición va última porque
acompaña a casi cualquier atasco: describe *cómo* se agotó el presupuesto,
no *por qué*.

La regla 1 tampoco es decorativa. Una corrida con credenciales vencidas
produjo 8 fallos que el clasificador aísla como `infra_error`: sin esa
regla, se contabilizarían como fallos del agente.

#### Resultados

Sobre los **9 fallos** de la corrida canónica:

| Categoría | Casos | % de fallos | Escenarios |
|---|---|---|---|
| `loop_estancado` | 6 | 67 % | vault-combination (2), backtracking-vault (2), color-locks, office-sequence |
| `terminacion_prematura` | 2 | 22 % | color-locks, library-search |
| `loop_navegacion` | 1 | 11 % | vault-combination |

Una traza de ejemplo por categoría, con la evidencia que emite el arnés:

| Escenario | Categoría | Evidencia |
|---|---|---|
| backtracking-vault | `loop_estancado` | 35/40 llamadas repetidas (88 %) hasta agotar el presupuesto |
| library-search | `terminacion_prematura` | el bucle cerró solo tras 13 llamadas, sin agotar el presupuesto |
| vault-combination | `loop_navegacion` | 14/54 llamadas `go` devolvieron error (26 %) |

#### Qué implica cada modo

**`loop_estancado` (67 %) — el fallo dominante, y es nuestro.** El agente
no se pierde ni alucina: repite entre el 61 % y el 88 % de sus llamadas
hasta quedarse sin presupuesto. Sabe qué hacer paso a paso, pero **no
lleva registro de lo que ya hizo**. Es una carencia del framework, no del
modelo: un agente con memoria explícita de acciones ejecutadas (o una
comprobación de no-op antes de invocar) evitaría la mayoría de estos
casos. Es la mejora de mayor impacto que identificamos.

**`terminacion_prematura` (22 %) — también nuestra, y más grave.** En
`library-search` el agente hizo 13 llamadas (1 `look`, 11 `examine`, 1
`use`), no tomó ningún objeto y **cerró el bucle a mitad de camino**,
declarándolo en su respuesta final: *"No tengo ninguna otra idea de cómo
salir de la sala"*. Lo notable es que ninguna métrica de proceso lo
detecta: 0 errores, dentro de presupuesto y **rúbrica 8/8, el máximo
posible**. Por todo indicador de comportamiento se ve impecable, y falló
por completo.

Es un fallo de **condición de terminación**: el bucle de M1 corta cuando
el LLM devuelve texto sin `tool_calls`, así que basta con que el modelo se
declare sin ideas para que el agente dé el problema por cerrado. Un cierre
condicionado a verificación de meta —o un reintento con contexto de que el
objetivo sigue pendiente— lo evitaría. Es el caso que mejor justifica por
qué la métrica de éxito se comprueba sobre el **estado del mundo** y no
sobre el texto del agente (§2.2).

**`loop_navegacion` (11 %) — realimentación desaprovechada.** El agente
insiste con direcciones inválidas mientras la herramienta le responde
literalmente *"Salidas disponibles: sur"*. El dato correcto está en el
historial y no se usa.

**Lo que el desglose descarta.** Ningún fallo fue `id_alucinado` ni
`infra_error` en esta corrida: el agente **no inventa objetos** y la
infraestructura no falló una sola vez en 24 ejecuciones. Descartar causas
es tan informativo como confirmarlas: el problema no es la percepción del
entorno sino la gestión del propio progreso.

#### Calibración, y una corrección de método

Los umbrales de las reglas 3–5 se fijaron sobre una corrida de 8 casos
(4 fallos), donde cada uno caía en un hueco vacío de los datos con márgenes
de 10×, 6.5× e ∞. Con 4 fallos la repetición de llamadas parecía **no**
discriminar —`apartment-keys` repetía 78 % y resolvía el escenario—, así
que se medía como evidencia pero no clasificaba.

Al aplicar esa taxonomía a las 24 corridas, **6 de 9 fallos cayeron en el
fallback**: dos tercios sin diagnóstico. Y los seis repetían ≥61 %,
mientras los casos resueltos median 29 % (máximo 53 %, con un único caso
en 65 %). Con más datos, la señal que habíamos descartado sí separaba.

De ahí la regla 6, con el umbral en el hueco 53 %–61 %. Tras recalibrar, el
fallback quedó **vacío**: los 9 fallos tienen causa asignada. La
reclasificación no requirió volver a ejecutar el agente — el flag
`--reclassify` reprocesa las trazas ya guardadas en segundos y sin gastar
tokens.

Registramos esto como parte del método, no como una nota al pie: **una
taxonomía calibrada con 4 observaciones no generalizó a 24**, y sólo se
detectó porque el arnés permite validar sobre corridas repetidas.

**Limitaciones que asumimos.**

- El umbral de `loop_estancado` (55 %) tiene un **margen estrecho** —separa
  53 % de 61 %— comparado con los de las reglas 3–5. Es el candidato a
  revisar con más corridas.
- **Tres categorías no dispararon** en esta corrida (`id_alucinado`,
  `loop_improductivo`, `infra_error`). Las tres sí dispararon en corridas
  anteriores, pero ninguna corrida sola ejercita las siete reglas.
- La taxonomía clasifica **una causa por caso**. Cuando dos señales
  coinciden, la precedencia decide y la otra queda sólo en las señales del
  registro, no en la categoría.
- Las reglas dependen de las **cadenas de texto exactas** que emiten las
  world-tools. Un cambio de redacción en `mia_world` degradaría el
  diagnóstico en silencio; por eso están fijadas como constantes en los
  tests (`tests/student/test_failure_analysis.py`).

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

### 4.3. Presupuesto de iteraciones `max_iterations` (40 → 10)

**Qué se cambió.** Única variable: el tope de iteraciones del bucle del
agente — 40 (base US-02) vs 10 — con todo lo demás igual (prompt v1,
nova-lite). Para controlar la varianza del modelo (§5) se corrió el
subconjunto **medium+hard** (4 escenarios) con **3 repeticiones** por
escenario usando el nuevo flag `--repeat 3` (12 corridas por config).

| Escenario | Dif. | Éxito \@40 | Éxito \@10 | Calls medio \@40 | Rúbrica \@40 / \@10 |
|---|---|---|---|---|---|
| color-locks | medium | 1/3 (0.33) | 0/3 | 33.7 | 5.0 / 5.0 |
| apartment-keys | medium | 3/3 (1.0) | 0/3 | 13.3 | 5.7 / 6.0 |
| library-search | hard | 2/3 (0.67) | 0/3 | 24.7 | 6.7 / 7.0 |
| office-sequence | hard | 1/3 (0.33)\* | 0/3 | 26.7 | 3.7 / 5.7 |
| **Total** | | **7/12 (0.58)** | **0/12 (0.0)** | | |

\* office-sequence \@40 tuvo 1 corrida con `ModelErrorException` (fallo
transitorio de nova-lite en la secuencia de tool-use), contada como
no-resuelta. Costos por config: latencia media 22–28 s \@40 vs ~11 s \@10;
tokens de entrada ~540–700k \@40 vs ~165–185k \@10 (por 6 corridas).

**Conclusión.** Recortar el presupuesto de 40 a 10 iteraciones **colapsa
el éxito de 58 % a 0 %** en medium/hard: ninguna cadena se completa. La
causa **no** es la calidad del proceso — la rúbrica media a 10 iteraciones
es igual o **mayor** (p. ej. library-search 7.0 vs 6.7) porque se acumulan
menos acciones redundantes — sino que el agente **se queda sin presupuesto
antes de terminar**: la exploración (`look`/`examine`) consume iteraciones
antes de las acciones, y estos escenarios necesitan 7–13 llamadas óptimas
más el margen de exploración. Esto aísla `max_iterations` como parámetro de
infraestructura **necesario pero no suficiente**: 40 habilita las
soluciones pero no las garantiza (la varianza del modelo sigue mandando).
Es el complemento de §4.2 — el prompt guía *qué* hacer; el presupuesto
define *cuánto* margen hay para hacerlo.

### 4.4. Experimentos propuestos (no ejecutados)

- **Memoria acotada (`max_history_messages` chico):** impacto en escenarios
  multi-sala que exigen recordar el mapa.

## 5. Limitaciones y próximos pasos

- **Varianza del LLM (limitación principal de la métrica).** La única
  fuente de no-determinismo del arnés es el modelo (temperatura 0.2, sin
  seed). **El pass/fail de una sola corrida no es confiable**, y la
  corrida canónica lo cuantifica: 4 de los 8 escenarios tienen tasas
  intermedias (1/3 o 2/3), es decir que cambian de resultado entre
  ejecuciones idénticas. `color-locks` y `library-search` figuraban como
  resueltos o fallados en corridas anteriores según la tirada. Por eso
  todos los resultados principales se reportan sobre `--repeat 3`
  (pass@k) con tabla de estabilidad. Próximo paso: reportar también el
  **desvío estándar** y subir N donde el presupuesto de tokens lo permita.
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
- **Punto ciego de la rúbrica: los loops de período corto.** El análisis
  de errores destapó una debilidad de nuestra propia métrica cualitativa.
  `_score_error_recovery` compara cada paso fallido **sólo con el
  inmediatamente siguiente**: si son distintos, cuenta como recuperación.
  Un agente atrapado alternando `este`/`oeste` contra dos salidas
  inválidas obtiene por eso **2/2 en "recuperación de errores"**, pese a
  26 llamadas fallidas seguidas. La rúbrica premia el cambio de acción,
  no el progreso. El arreglo es evaluar una ventana de k pasos en lugar de
  uno, detectando ciclos en vez de repeticiones inmediatas. Lo dejamos
  documentado y no corregido: cambiar la rúbrica a esta altura invalidaría
  la comparabilidad de §4.2 y §4.3, que ya la usan como medida.
- **La rúbrica tampoco detecta el abandono.** El caso de
  `terminacion_prematura` en `library-search` puntúa **8/8** —el máximo— y
  falla por completo (§3.3). Ninguna de las cuatro dimensiones pregunta si
  el agente *avanzó hacia la meta*: miden higiene de proceso, no progreso.
  Una quinta dimensión de avance (p. ej. fracción de sub-objetivos
  alcanzados) cerraría ese hueco.
- **Taxonomía de fallos calibrada con pocos datos.** Los umbrales de
  `classify_failure` se ajustaron sobre 4 fallos y no generalizaron a 24
  (§3.3); tras recalibrar, el umbral de `loop_estancado` conserva un margen
  estrecho (53 %–61 %). Con más corridas habría que revisarlo, y el flag
  `--reclassify` permite hacerlo sin volver a gastar tokens.
