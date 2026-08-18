# Milestone 3 — Evaluación sobre un problema objetivo

## Objetivo

Usen su framework para resolver un problema concreto asignado, evalúenlo
con rigor y expliquen lo que encontraron.


## El problema: sala de escape en un mundo simulado

Su agente recibe acceso a un **mundo simulado tipo sala de escape**
(inspirado en ALFWorld) y debe encontrar la forma de abrir la puerta
principal de la sala en la que se encuentra. El mundo es fijo y vive en
`scaffold/mia_world/` es parte del repositorio del TP.

**Herramientas disponibles.** Verbos genéricos en `mia_world/tools.py`,
listos para registrar con `agent.register_tool(...)`. Los cuatro primeros
están en todos los escenarios; `go` solo aparece en los escenarios con
varias salas (los de una sola sala no lo registran):

| Verbo | Args | Para qué sirve |
|---|---|---|
| `look` | — | Describe la sala: items visibles, estado de puertas, salidas e inventario. |
| `examine` | `target` | Inspecciona un objeto; los contenedores revelan su contenido. |
| `take` | `item` | Coge un objeto al inventario. |
| `use` | `item`, `target` | Aplica un objeto del inventario sobre otro de la sala (típicamente, una llave sobre una cerradura). |
| `go` | `direction` | (Solo multi-sala) Navega por una salida hacia la sala contigua. |

**Dataset (`scaffold/scenarios/`).** Escenarios de dificultad creciente.
La meta se comprueba con `mia_world.check_goal` sobre el estado del mundo
(no sobre el texto del agente; esto da una métrica fiable). La mayoría
busca `puerta_principal.open_state == "open"`, pero los escenarios de
navegación introducen goals adicionales (ver más abajo).

| Dificultad | Escenario | Mecánica | Optimal | Brute-force peor caso |
|---|---|---|---:|---:|
| `easy` | `study-with-key` | Llave bajo una alfombra | 3 calls | 3 calls |
| `medium` | `color-locks` | Cadena de cofres con llaves de colores | 11 calls | 11 calls |
| `medium` | `apartment-keys` | **Multi-sala**: navegar 3 ambientes para hallar la llave y volver | 7 calls | 7 calls |
| `hard` | `library-search` | 1 de 8 libros + caja fuerte intermedia | 7 calls | 13 calls |
| `hard` | `office-sequence` | **Multi-sala + goal compuesto/ordenado**: recuperar el documento *antes* de abrir la puerta | 13 calls | 13 calls |
| `extreme` | `extreme-archive` | 1 de 20 expedientes con prosa burocrática (~16 K tokens) | 4 calls | **No cabe en 16 K tokens de contexto** |
| `extreme` | `vault-combination` | **Multi-sala + cerradura multi-item**: combinar 3 núcleos de 3 salas (con llaves encadenadas entre salas y una puerta gated) | 21 calls | 21 calls |
| `extreme` | `backtracking-vault` | **Backtracking profundo**: el cofre de la 1ª sala solo abre con la llave de la última; 2 puertas gated en el camino | 18 calls | 18 calls |

**Mecánicas de los escenarios `extreme` de horizonte largo.** Dos campos
opcionales del motor habilitan dependencias profundas:

- **Cerradura multi-item** (`locked: {"requires_items": [...]}`): el Item
  solo abre tras `use` de *todas* las piezas indicadas (en cualquier orden).
  Fuerza a *combinar* objetos hallados en salas distintas.
- **Salida con puerta** (`Room.locked_exits: {dirección: item_puerta}`):
  `go` solo cruza si ese Item-puerta está `open`. Habilita progresión gated
  y backtracking (volver a una sala anterior con algo del final).

**Escenarios multi-sala (`medium`/`hard`).** A partir de `medium` algunos
escenarios definen `exits` por sala y registran el verbo `go`. Esto añade
presión arquitectónica nueva sobre el agente:

- `apartment-keys` (medium) — la llave está en otra sala: el agente debe
  **navegar, recordar el mapa y volver** a la sala de la puerta. Pone a
  prueba la memoria de estado del M2 de forma que un único `look` no basta.
- `office-sequence` (hard) — la meta es un goal **compuesto y ordenado**
  (`sequence`): hay que tener el `documento_confidencial` en el inventario
  **antes** de abrir `puerta_principal` (que "se sella" al abrirse). Esto
  premia un agente que **descompone y planifica** el orden de sub-objetivos
  en lugar de reaccionar paso a paso, y abre la puerta a un experimento
  *planner explícito vs ReAct puro*.

Tipos de goal soportados por `check_goal`: hojas (`item_open`,
`agent_in_room`, `item_in_inventory`) y combinadores (`all_of`, `any_of`,
`sequence`). El orden de `sequence` se verifica con el `event_log` del
mundo, que las herramientas pueblan automáticamente.

**El escenario `extreme` está diseñado para no caber.** Si su agente
intenta examinar los veinte expedientes en su contexto principal, supera
la ventana de la mayoría de modelos pequeños y empieza a perder disciplina
de tool-calling.

**Runner de escenarios.** Ya disponible en `scaffold/mia_world/cli.py`:

```bash
python -m mia_world.cli list                # muestra los 4 escenarios
python -m mia_world.cli run --scenario easy # ejecuta uno end-to-end
```

## Lo que deben construir

- **Una infraestructura de evaluación.** Ejecuten su agente sobre el
  dataset, capturen entradas/salidas/llamadas a herramientas/errores por
  caso y produzcan un informe resumen. La infraestructura debe ser
  reproducible (`python eval/run.py`) sin pasos manuales.
- **Métricas.** Al menos una métrica cuantitativa (accuracy, exact match,
  F1, pass@k, latencia, coste — la que mejor encaje con el problema) y al
  menos una dimensión cualitativa evaluada vía rúbrica o LLM-as-judge.
  Justifiquen ambas elecciones.
- **Análisis de errores.** Un desglose de dónde y por qué falla su agente.
  Categoricen los modos de fallo; no se limiten a un único número.
- **Experimentos.** Al menos dos experimentos que muestren *qué partes de su
  framework importan para este problema*. Ejemplos: apagar el resumen,
  intercambiar la estrategia de prompting, sustituir una herramienta por
  un no-op, reducir max steps.

## Informe obligatorio

1. Aproximación — cómo aplicaron su framework de M1+M2 al problema y qué
   (si algo) especializaron.
2. Métricas — qué midieron, por qué y cómo lo computaron.
3. Resultados — números principales y desglose por categoría.
4. Experimentos — qué cambiaron, qué pasó, qué concluyeron.
5. Limitaciones y qué construirían a continuación.
