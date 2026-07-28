# Milestone 2: Memoria, prompting y robustez

## Objetivo

Hagan que el framework sobreviva a conversaciones largas, salidas
malformadas del modelo y fallos transitorios. Agreguen manejo de errores
a las herramientas y reporten el consumo de tokens de cada ejecución.

## Alcance

M2 se construye sobre M1. Mantengan la misma fachada externa
(`build_agent`, `register_tool`, `run`) y agreguen comportamiento
conversacional, validación y resiliencia dentro del agente.

> **Importante:** Toda la lógica de M2 (memoria, salida estructurada,
> reintentos, prompting) debe vivir en el agente, no en el cliente LLM.
> De la misma forma que funcionó en M1. Esto es necesario para que
> podamos correr los tests automatizados contra su implementación.

## Lo que deben construir

- **Estado conversacional (statefulness).** En M1 cada llamada a `run`
  se trataba como una interacción independiente. En M2, el agente tiene
  estado: llamadas sucesivas a `run` sobre la misma instancia continúan
  la misma conversación.
- **Gestión de contexto / memoria.** El constructor del agente acepta
  `max_history_messages`. La lista de mensajes que envían al cliente LLM
  en cada llamada a `chat(...)` nunca puede superar ese tope, sin
  importar cuántos turnos lleve la conversación. La estrategia
  obligatoria es *sliding window*; pueden decidir qué mensajes conservar
  y cuáles descartar (justifiquen). También pueden implementar
  estrategias alternativas, como *offload/retrieve* o *summarization*;
  justifiquen su diseño y tradeoffs.
- **Recencia.** Cualquiera sea la estrategia, hay una invariante que no
  pueden romper: el **mensaje de usuario más reciente** siempre debe
  aparecer en la siguiente llamada al LLM. Pueden descartar contexto
  antiguo para respetar el presupuesto, pero nunca lo que el usuario
  acaba de decir.
- **Resiliencia del historial.** El agente debe sobrevivir a
  conversaciones largas (decenas de turnos con mensajes grandes) sin
  romperse ni degradar la respuesta: cada `run` sigue devolviendo un
  `AgentResult` con `answer` no vacío.
- **Manejo de salida estructurada.** `structured_call(...)` debe exigir
  al LLM una respuesta estructurada mediante la herramienta sintética
  `final_result` (nombre fijo, ver `mia_agents.tool_schema`). El agente
  ofrece al LLM un schema de Pydantic, valida los argumentos del
  `tool_call` a `final_result` y reintenta con un prompt de reparación
  si la validación falla o si el modelo responde con texto libre en
  lugar de invocar la tool. Definan un número máximo de reintentos y una
  estrategia de fallo cuando no se pueda llegar al formato deseado.
- **Errores recuperables en herramientas de M1.** Mejoren el manejo de
  errores de las herramientas de M1 (calculadora simple y lector de
  archivos). La idea no es solo "no crashear", sino distinguir qué
  fallos son recuperables (el LLM puede corregir los argumentos e
  intentar de nuevo) y devolver un mensaje accionable en el resultado de
  la herramienta.

  **Calculadora** — ejemplos de errores recuperables y respuesta útil:
  - Operandos que no son numéricos: indicar qué parámetro falló, qué
    valor recibió y por qué no es válido.
  - Operador no soportado: listar los operadores permitidos (`+`, `-`,
    `*`, `%`).
  - División o módulo por cero: explicar la restricción sin un mensaje
    genérico.

  **Lector de archivos** — ejemplos de errores recuperables y respuesta
  útil:
  - Ruta vacía, absoluta, con `..` o que escapa del sandbox: explicar la
    regla violada y cómo debe verse una ruta válida.
  - Archivo inexistente: si el directorio contenedor existe y es
    válido, listar los archivos disponibles ahí para que el LLM pueda
    elegir la ruta correcta.
  - La ruta apunta a un directorio en lugar de un archivo: indicarlo y,
    si aplica, listar el contenido del directorio.
- **Resiliencia.** El agente envuelve sus llamadas al cliente LLM y a
  las herramientas de forma que los fallos transitorios (timeouts,
  5xx, rate limits, excepciones de red) se reintenten y los demás
  errores afloren limpios.
- **Tracking de tokens.** `AgentResult.input_tokens` y
  `AgentResult.output_tokens` deben sumar los tokens reportados por el
  cliente LLM durante una llamada a `run(...)`.

## Informe

- **Estrategia de memoria:** cómo fue implementada y qué problemas
  encontraron. Si implementan una estrategia alternativa a sliding
  window, comenten por qué la eligieron y qué tradeoffs tiene.
- **Salida estructurada:** cómo ofrecen `final_result` al LLM, cómo
  validan los argumentos, cómo se reparan los fallos de validación, qué
  pasa cuando se agotan los reintentos.
- **Errores en herramientas:** qué errores recuperables detectan la
  calculadora y el lector de archivos, qué información devuelven al LLM
  para facilitar la corrección, y un ejemplo concreto de recuperación en
  cada una.
- **Modos de fallo** que dejaron deliberadamente dentro vs. fuera del
  alcance.

## Criterios de aprobación

- [ ] Pueden ejecutar una conversación que supere su presupuesto de
      contexto y el agente sigue comportándose con sensatez.
- [ ] Un prompt de salida estructurada deliberadamente roto en su suite
      de tests dispara el flujo de reparación y bien se recupera, o bien
      falla limpiamente.
- [ ] Un fallo transitorio simulado (e.j. un timeout del cliente LLM) se
      reintenta y la ejecución termina con éxito.
- [ ] La calculadora y el lector de archivos devuelven mensajes claros y
      accionables ante errores recuperables.
- [ ] `AgentResult.input_tokens` y `AgentResult.output_tokens` reflejan
      correctamente los tokens reportados por el cliente LLM.
- [ ] Informe con las secciones descriptas previamente.
