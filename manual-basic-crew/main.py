from crewai import Agent, Task, Crew, Process, LLM


# 1. Modelo local con Ollama
llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.2
)


# 2. Agente investigador
researcher = Agent(
    role="Investigador técnico",
    goal="Analizar claramente el tema: {topic}",
    backstory=(
        "Eres un investigador técnico. "
        "Tu trabajo es analizar un tema, identificar ideas importantes, "
        "ventajas, riesgos y casos de uso prácticos."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# 3. Agente redactor
writer = Agent(
    role="Redactor ejecutivo",
    goal="Convertir análisis técnicos en resúmenes claros y útiles",
    backstory=(
        "Eres un redactor ejecutivo. "
        "Tu trabajo es tomar un análisis técnico y convertirlo en "
        "un resumen fácil de presentar en una reunión."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)


# 4. Tarea del investigador
research_task = Task(
    description=(
        "Analiza el tema: {topic}. "
        "Incluye definición breve, ventajas, riesgos, casos de uso "
        "y una conclusión práctica."
    ),
    expected_output=(
        "Un análisis estructurado con definición, ventajas, riesgos, "
        "casos de uso y conclusión."
    ),
    agent=researcher
)


# 5. Tarea del redactor
summary_task = Task(
    description=(
        "Usa el análisis anterior para crear un resumen ejecutivo. "
        "Debe ser claro, breve y útil para explicar el tema en una reunión."
    ),
    expected_output=(
        "Un resumen ejecutivo con máximo 5 puntos principales."
    ),
    agent=writer,
    context=[research_task]
)


# 6. Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, summary_task],
    process=Process.sequential,
    verbose=True
)


# 7. Ejecutar
result = crew.kickoff(
    inputs={
        "topic": "uso de agentes de inteligencia artificial en recursos humanos"
    }
)


print("\n\n===== RESULTADO FINAL =====\n")
print(result)