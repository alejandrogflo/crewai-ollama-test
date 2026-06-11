from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    from .tools import SUPPORTED_DOCUMENT_EXTENSIONS, discover_candidate_files, is_supported_document
    from .workflow import build_recruiting_workflow
except ImportError:
    from tools import SUPPORTED_DOCUMENT_EXTENSIONS, discover_candidate_files, is_supported_document
    from workflow import build_recruiting_workflow


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    default_inputs_dir = project_dir / "inputs"
    parser = argparse.ArgumentParser(
        description="Run a local CrewAI recruiting workflow on a job description and candidate CVs.",
    )
    parser.add_argument(
        "--job",
        default=str(project_dir / "data" / "jobs" / "ai_hr_analyst.md"),
        help="Path to the job description file (.md, .txt, .pdf).",
    )
    parser.add_argument(
        "--candidates-dir",
        default=str(project_dir / "data" / "candidates"),
        help="Directory containing candidate files (.md, .txt, .pdf).",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Individual candidate file path. Repeat this flag to add more than one CV.",
    )
    parser.add_argument(
        "--model",
        default="ollama/llama3.1:8b",
        help="LiteLLM model identifier for Ollama.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Base URL for your Ollama server.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the local model.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=2,
        help="How many candidates to include in the shortlist.",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_dir / "outputs"),
        help="Parent directory where each run folder will be created.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce CrewAI verbosity.",
    )
    parser.add_argument(
        "--init-inputs",
        action="store_true",
        help="Create a ready-to-fill inputs folder for a real run and exit.",
    )
    parser.add_argument(
        "--inputs-dir",
        default=str(default_inputs_dir),
        help="Base folder used by --init-inputs.",
    )
    return parser.parse_args()


def build_run_directory(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_directory = output_root / timestamp
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_document_path(path: str | Path, label: str) -> Path:
    document_path = Path(path).expanduser().resolve()
    if not document_path.exists():
        raise FileNotFoundError(f"{label} not found: {document_path}")
    if not is_supported_document(document_path):
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise ValueError(f"{label} must be one of: {supported}. Got: {document_path}")
    return document_path


def collect_candidate_paths(directory: str | Path | None, candidates: Iterable[str]) -> list[Path]:
    collected: list[Path] = []

    if directory:
        directory_path = Path(directory).expanduser().resolve()
        if directory_path.exists():
            if directory_path.is_dir():
                collected.extend(discover_candidate_files(directory_path))
            else:
                raise NotADirectoryError(
                    f"Candidate directory is not a directory: {directory_path}"
                )
        elif not list(candidates):
            raise FileNotFoundError(f"Candidate directory not found: {directory_path}")

    for candidate in candidates:
        collected.append(validate_document_path(candidate, "Candidate file"))

    deduped = sorted({path.resolve() for path in collected})
    if not deduped:
        raise FileNotFoundError(
            "No candidate files were found. Use --candidates-dir or one or more --candidate paths."
        )
    return deduped


def initialize_inputs(inputs_dir: Path) -> None:
    job_dir = inputs_dir / "job"
    candidates_dir = inputs_dir / "candidates"
    job_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    readme_path = inputs_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "\n".join(
                [
                    "# Recruiting Crew Inputs",
                    "",
                    "1. Put your job description in `job/` as `.md`, `.txt`, or `.pdf`.",
                    "2. Put your CVs in `candidates/` as `.md`, `.txt`, or `.pdf`.",
                    "3. Run:",
                    "",
                    "```bash",
                    "./venv/bin/python recruiting_crew/main.py \\",
                    f"  --job {job_dir / 'job_description.md'} \\",
                    f"  --candidates-dir {candidates_dir}",
                    "```",
                ]
            ),
            encoding="utf-8",
        )

    sample_job = job_dir / "job_description.md"
    if not sample_job.exists():
        sample_job.write_text(
            "\n".join(
                [
                    "# Replace This With Your Job Description",
                    "",
                    "Role title:",
                    "",
                    "Mission:",
                    "",
                    "Must-have requirements:",
                    "",
                    "Nice-to-have requirements:",
                    "",
                    "What success looks like in 90 days:",
                ]
            ),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    if args.init_inputs:
        inputs_dir = Path(args.inputs_dir).expanduser().resolve()
        initialize_inputs(inputs_dir)
        print(f"Inputs folder ready at: {inputs_dir}")
        print(f"Put the job description in: {inputs_dir / 'job'}")
        print(f"Put candidate CVs in: {inputs_dir / 'candidates'}")
        return

    job_path = validate_document_path(args.job, "Job description")
    candidate_paths = collect_candidate_paths(args.candidates_dir, args.candidate)
    top_n = max(1, min(args.top_n, len(candidate_paths)))
    run_directory = build_run_directory(Path(args.output_root).expanduser().resolve())

    workflow_result = build_recruiting_workflow(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        job_path=job_path,
        candidate_paths=candidate_paths,
        top_n=top_n,
        output_dir=run_directory,
        verbose=not args.quiet,
    )

    task_summaries = []
    for task_output in workflow_result.task_records:
        task_summaries.append(
            {
                "name": task_output.name,
                "agent": task_output.agent,
                "output_format": task_output.output_format,
                "raw": task_output.raw,
                "structured": task_output.structured,
                "attempts": task_output.attempts,
                "validation_errors": task_output.validation_errors,
                "fallback_used": task_output.fallback_used,
            }
        )

    save_json(
        run_directory / "run_summary.json",
        {
            "job_path": str(job_path),
            "candidate_paths": [str(path) for path in candidate_paths],
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "top_n": top_n,
            "token_usage": workflow_result.token_usage,
            "final_output": workflow_result.markdown_report,
            "tasks": task_summaries,
        },
    )

    print("\n===== SHORTLIST REPORT =====\n")
    print(workflow_result.markdown_report)
    print(f"\nOutputs saved to: {run_directory}")


if __name__ == "__main__":
    main()
