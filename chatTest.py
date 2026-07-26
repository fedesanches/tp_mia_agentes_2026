import textwrap

from student_framework import build_agent

agent = build_agent()  # sin mock: usa Ollama real vía LLMClient.from_env()

ANCHO = 80

print("Chat con el agente (Ctrl+C para salir)")
print("=" * ANCHO)

turno = 0
while True:
    turno += 1
    msg = input(f"\n[{turno}] vos: ")
    result = agent.run(msg)

    respuesta = textwrap.fill(
        result.answer or "",
        width=ANCHO,
        initial_indent="      ",
        subsequent_indent="      ",
    )
    print(f"[{turno}] agente:")
    print(respuesta)
    print("-" * ANCHO)
