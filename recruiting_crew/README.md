# Recruiting Crew

Este experimento toma la demo secuencial inicial y la mueve hacia un caso más útil para CrewAI: analizar una vacante y varios candidatos para producir una shortlist.

## Qué explora

- Orquestación multiagente con varias tareas dependientes.
- Salidas estructuradas con Pydantic.
- Validación semántica antes de pasar de una etapa a la siguiente.
- Tools locales para leer archivos Markdown, TXT y PDF.
- Generación de artefactos en `outputs/` para revisar cada corrida.

## Estructura

- `main.py`: CLI para ejecutar el flujo.
- `workflow.py`: agentes y tareas CrewAI.
- `schemas.py`: contratos estructurados.
- `validators.py`: validación semántica contra el texto fuente.
- `tools.py`: tools y helpers locales.
- `data/`: vacante y candidatos de ejemplo.

## Cómo correrlo

1. Asegúrate de tener Ollama corriendo en `http://localhost:11434`.
2. Ten disponible el modelo local, por ejemplo `llama3.1:8b`.
3. Ejecuta:

```bash
./venv/bin/python recruiting_crew/main.py
```

Si quieres preparar una carpeta para insumos reales:

```bash
./venv/bin/python recruiting_crew/main.py --init-inputs
```

También puedes apuntarlo a tus propios archivos:

```bash
./venv/bin/python recruiting_crew/main.py \
  --job /ruta/a/vacante.md \
  --candidates-dir /ruta/a/candidatos \
  --model ollama/llama3.1:8b \
  --top-n 3
```

También puedes mezclar carpeta y archivos sueltos:

```bash
./venv/bin/python recruiting_crew/main.py \
  --job recruiting_crew/inputs/job/job_description.md \
  --candidates-dir recruiting_crew/inputs/candidates \
  --candidate /ruta/otro_cv.pdf
```

## Qué genera

Cada corrida crea una carpeta en `recruiting_crew/outputs/<timestamp>/` con:

- `shortlist_report.md`: memo final en Markdown.
- `run_summary.json`: resumen estructurado de la corrida y cada tarea.
- `crew_execution.log`: log de ejecución.

## Siguientes ideas

- Añadir un agente revisor que cuestione puntajes débiles.
- Separar extracción y scoring por lotes para medir latencia.
- Guardar memoria de candidatos procesados.
- Agregar una interfaz web o un panel simple para comparar corridas.
- Incluir parsing de PDF reales y tests con CVs anonimizados.
