# Recruiting Crew Inputs

1. Put your job description in `job/` as `.md`, `.txt`, or `.pdf`.
2. Put your CVs in `candidates/` as `.md`, `.txt`, or `.pdf`.
3. Run:

```bash
./venv/bin/python recruiting_crew/main.py \
  --job /Users/macdealejandro/pps/crewai-ollama-test/recruiting_crew/inputs/job/job_description.md \
  --candidates-dir /Users/macdealejandro/pps/crewai-ollama-test/recruiting_crew/inputs/candidates
```