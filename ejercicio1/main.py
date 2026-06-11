from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(
    model="ollama/llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.2
)

# ****** Agents ******
analista_perfil = Agent(
    role="Experto reclutador de perfiles para puestos en una empresa",
    goal="Analizar el perfil del candidato e identificar sus capacidades, experiencia y fortalezas relevantes",
    backstory=(
        "Eres un experto analista reclutador "
        "Tu trabajo es extraer la informacion mas relevante de los candidatos "
        "fortalezas, experiencia, capacidades, estudios y toda la informacion relevante del perfil."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

evaluador_tecnico = Agent(
    role="Experto evaluador tecnico de prospectos para perfiles",
    goal="Evaluar si el candidato encaja con el puesto solicitado: {job_position}",
    backstory=(
        "Eres un experto evaluador técnico. "
        "Tu trabajo es comparar las habilidades del candidato contra los requisitos del puesto "
        "e identificar coincidencias, brechas y nivel de ajuste."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

redactor = Agent(
    role="Experto redactor de recomendaciones",
    goal="Generar una recomendacion clara del prospecto adecuado para el puesto",
    backstory=(
        "Eres un experto redactor. "
        "Tu trabajo es redactar una recomendación clara, breve y justificada "
        "sobre el nivel de encaje entre el candidato y el puesto solicitado."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

####  ******  Tasks ******

analizar_perfil_task = Task(
    description=(
        "Analiza el perfil del candidato: {candidate_profile}. "
        "Identifica experiencia relevante, fortalezas, debilidades y posibles alertas."
    ),
    expected_output=(
        "Un análisis estructurado con experiencia relevante, fortalezas, debilidades, "
        "posibles alertas y resumen profesional del candidato."
    ),
    agent=analista_perfil
)

comparador_perfil_puesto = Task(
    description=(
        "Compara el análisis del candidato con el puesto solicitado: {job_position}. "
        "Evalúa coincidencias entre las capacidades del candidato y los requisitos del puesto. "
        "Identifica brechas técnicas, fortalezas relevantes y nivel de compatibilidad."
    ),
    expected_output=(
        "Una evaluación técnica con coincidencias, brechas, nivel de compatibilidad "
        "y justificación del ajuste entre candidato y puesto."
    ),
    agent=evaluador_tecnico, 
    context=[analizar_perfil_task] #La tarea técnica usará el resultado del análisis del perfil.
)

recomendacion = Task(
    description=(
        "Usa la evaluación anterior para generar una recomendación final. "
        "Clasifica al candidato como: Recomendado, Recomendado con reservas o No recomendado. "
        "Justifica la decisión con base en fortalezas, brechas y compatibilidad con el puesto."
    ),
    expected_output=(
        "Una recomendación final que incluya clasificación, justificación breve, "
        "principales fortalezas, principales brechas y decisión final."
    ),
    context=[comparador_perfil_puesto], #La recomendación final usará la evaluación técnica anterior.
    agent=redactor
)

crew = Crew(
    agents=[
        analista_perfil,
        evaluador_tecnico,
        redactor
    ],
    tasks=[
        analizar_perfil_task,
        comparador_perfil_puesto,
        recomendacion 
    ],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff(
    inputs={
        "candidate_profile": (
            "Luis Pérez tiene 2 años de experiencia como desarrollador Python. "
            "Ha trabajado con APIs REST, bases de datos PostgreSQL y automatización de procesos. "
            "Tiene conocimientos básicos de Docker, pero poca experiencia en despliegues en la nube. "
            "También ha trabajado en soporte técnico y documentación."
        ),
        "job_position": (
            "Desarrollador Backend Python con experiencia en APIs, bases de datos, Docker, "
            "servicios cloud y buenas prácticas de desarrollo."
        )
    }
)

print("\n\n===== RECOMENDACIÓN FINAL =====\n")
print(result)
