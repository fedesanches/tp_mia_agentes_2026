import textwrap

from student_framework import build_agent

agent = build_agent()  # sin mock: usa Ollama real vía LLMClient.from_env()

ANCHO = 80
MAX_TOOL_OUTPUT = 200


def _truncar(texto: str | None, largo: int = MAX_TOOL_OUTPUT) -> str:
    texto = texto or ""
    if len(texto) <= largo:
        return texto
    return texto[:largo] + "…"


print("Chat con el agente (Ctrl+C para salir)")
print("=" * ANCHO)

turno = 0
while True:
    turno += 1
    msg = input(f"\n[{turno}] vos: ")
    result = agent.run(msg)

    if result.steps:
        print(f"[{turno}] herramientas:")
        for paso in result.steps:
            estado = "ERROR" if paso.error else "ok"
            print(
                f"      {paso.tool_name}({paso.tool_input}) -> "
                f"{_truncar(paso.tool_output)}  [{estado}]"
            )

    respuesta = textwrap.fill(
        result.answer or "",
        width=ANCHO,
        initial_indent="      ",
        subsequent_indent="      ",
    )
    print(f"[{turno}] agente:")
    print(respuesta)
    print("-" * ANCHO)
