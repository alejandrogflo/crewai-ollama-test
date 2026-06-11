from crewai import Agent, Task, Crew, Process, LLM

# Modelo local de Ollama
llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.3
)

# Agente 1: Planificador
planner = Agent(
    role="Content Planner",
    goal="Crear un plan claro y útil sobre el tema: {topic}",
    backstory=(
        "Eres un planificador de contenido. "
        "Tu trabajo es organizar ideas, definir audiencia, "
        "proponer estructura y puntos clave."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Agente 2: Redactor
writer = Agent(
    role="Content Writer",
    goal="Escribir un artículo claro y bien estructurado sobre: {topic}",
    backstory=(
        "Eres un redactor profesional. "
        "Tomas el plan del planificador y lo conviertes en un artículo "
        "ordenado, claro y fácil de entender."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Agente 3: Editor
editor = Agent(
    role="Editor",
    goal="Revisar y mejorar el artículo final sobre: {topic}",
    backstory=(
        "Eres un editor. "
        "Corriges errores, mejoras claridad, estructura y tono."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Tarea 1: Planificar
plan_task = Task(
    description=(
        "Crea un plan de contenido sobre {topic}. "
        "Incluye: audiencia objetivo, objetivo del artículo, "
        "estructura con secciones y puntos clave."
    ),
    expected_output=(
        "Un plan de contenido con audiencia objetivo, objetivo, "
        "estructura del artículo y puntos clave."
    ),
    agent=planner
)

# Tarea 2: Escribir
write_task = Task(
    description=(
        "Usa el plan anterior para escribir un artículo sobre {topic}. "
        "Debe estar en formato Markdown, con título, introducción, "
        "secciones principales y conclusión."
    ),
    expected_output=(
        "Un artículo completo en Markdown, claro y bien estructurado."
    ),
    agent=writer,
    context=[plan_task]
)

# Tarea 3: Editar
edit_task = Task(
    description=(
        "Revisa el artículo anterior. Corrige gramática, mejora claridad, "
        "ordena las ideas y entrega una versión final lista para publicar."
    ),
    expected_output=(
        "Versión final del artículo en Markdown, corregida y mejorada."
    ),
    agent=editor,
    context=[write_task]
)

# Crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan_task, write_task, edit_task],
    process=Process.sequential,
    verbose=True
)

# Ejecutar
result = crew.kickoff(
    inputs={
        "topic": "cómo la inteligencia artificial puede ayudar en recursos humanos"
    }
)

print("\n\n===== RESULTADO FINAL =====\n")
print(result)