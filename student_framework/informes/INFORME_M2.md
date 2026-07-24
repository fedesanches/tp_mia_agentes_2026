# Informe Milestone 2

> Documento en progreso. Cada sección se completa a medida que se
> implementan las historias del backlog de M2.

## 1. Estrategia de memoria (US-01)

_Pendiente de implementación._

## 2. Salida estructurada con reparación (US-02)

_Pendiente de implementación._

## 3. Errores recuperables en herramientas (US-03)

En M1 las herramientas o crasheaban o devolvían mensajes genéricos. En M2
la calculadora y el lector de archivos distinguen los **errores
recuperables** (los que el LLM puede corregir cambiando sus argumentos y
reintentando) y devuelven, en lugar de una excepción, un `str` que empieza
con `Error:`, nombra el parámetro/valor que falló e indica cómo corregirlo.

Idea de diseño transversal: la herramienta nunca lanza una excepción hacia
el bucle del agente por una entrada inválida esperable. Devuelve un mensaje
accionable que el agente vuelca al historial como `role: "tool"`, de modo
que el LLM lo lea y se corrija en la siguiente iteración.

### 3.1. Calculadora (`student_framework/tools/calculator.py`)

| Error recuperable | Información que devuelve al LLM |
| --- | --- |
| Operando no numérico | El parámetro que falló (`left_operand` / `right_operand`), el valor exacto recibido, su tipo, y un ejemplo válido (`3` o `2.5`). |
| Operador no soportado | El operador inválido y la lista completa de permitidos: `'+', '-', '*', '%'`. |
| Módulo por cero | Que `right_operand` es 0 y que hay que pasar un divisor distinto de cero (mensaje específico, no genérico). |

Detalles de implementación:

- La validación de operandos vive en el helper `_coerce_operand(name, value)`,
  que devuelve `(numero, None)` en caso de éxito o `(None, mensaje)` con el
  error accionable.
- Acepta además números pasados como texto (`"3"` → `3.0`), un caso
  frecuente cuando el LLM serializa mal los argumentos.
- Rechaza `bool` como operando (aunque `bool` sea subclase de `int`), porque
  para una calculadora no es un número válido.

**Ejemplo concreto de recuperación:**

El LLM llama `calculator(left_operand="23a", right_operand=17, operator="*")`.
La herramienta devuelve:

```
Error: el parametro 'left_operand' debe ser numerico, pero recibio el texto
'23a', que no representa un numero. Pasa un numero, por ejemplo 3 o 2.5.
```

Con ese mensaje el LLM identifica exactamente qué argumento corregir y
reintenta `calculator(23, 17, "*")`, obteniendo `391.0`.

### 3.2. Lector de archivos (`student_framework/tools/file_reader.py`)

El lector define un **sandbox** cuya raíz es el directorio del proyecto
(`cwd`). Las rutas relativas se anclan a esa raíz; las rutas absolutas se
permiten como "rutas seguras" hacia archivos existentes.

| Error recuperable | Información que devuelve al LLM |
| --- | --- |
| Ruta vacía | Que la ruta está vacía y un ejemplo de ruta válida (`'datos/notas.txt'`). |
| Ruta con `..` | Que contiene navegación a directorios superiores (no permitido) y cómo se ve una ruta válida sin `..`. |
| Ruta que escapa del sandbox (p. ej. symlink) | Que la ruta sale del directorio permitido (muestra el root) y que use una ruta relativa dentro del proyecto. |
| Archivo inexistente | Si el directorio contenedor existe, **lista los archivos disponibles** ahí para que el LLM elija la ruta correcta. |
| La ruta es un directorio | Que apunta a un directorio, no a un archivo, y **lista los archivos** dentro de ese directorio. |
| Extensión no permitida | La extensión recibida y la lista de extensiones válidas. |
| Archivo demasiado grande | Que superó el tope de tamaño (`_MAX_BYTES`). |
| Archivo no UTF-8 | Que no es texto UTF-8 válido; solo se leen archivos de texto. |

Orden de validación (de más barato/estructural a más costoso): ruta vacía →
`..` → escape del sandbox → inexistencia → directorio → extensión → lectura
(tamaño/UTF-8). Ese orden permite dar el mensaje más útil primero (por
ejemplo, listar archivos disponibles antes de quejarse de la extensión).

**Ejemplo concreto de recuperación:**

El LLM llama `file_reader(path="notas.txt")` pero el archivo real se llama
`numero.txt`. La herramienta devuelve:

```
Error: el archivo 'notas.txt' no existe. En el directorio 'C:\...\proyecto'
hay estos archivos disponibles: numero.txt, README.md, ...
```

El LLM ve el nombre correcto en la lista y reintenta
`file_reader(path="numero.txt")`, obteniendo el contenido.

## 4. Resiliencia ante fallos transitorios (US-04)

_Pendiente de implementación._

## 5. Tracking de tokens (US-05)

_Pendiente de implementación._

## 6. Modos de fallo dentro / fuera de alcance

_Pendiente de completar a medida que se cierren las historias._
