from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task

try:
    from .schemas import (
        CandidateAssessment,
        CandidateProfile,
        JobScorecard,
        ShortlistReport,
    )
    from .tools import read_document_text
    from .validators import (
        compute_weighted_score,
        recommendation_from_score,
        validate_assessment,
        validate_profile,
        validate_scorecard,
        validate_shortlist,
    )
except ImportError:
    from schemas import CandidateAssessment, CandidateProfile, JobScorecard, ShortlistReport
    from tools import read_document_text
    from validators import (
        compute_weighted_score,
        recommendation_from_score,
        validate_assessment,
        validate_profile,
        validate_scorecard,
        validate_shortlist,
    )


@dataclass
class TaskRunRecord:
    name: str
    agent: str
    output_format: str
    raw: str
    structured: dict[str, Any]
    attempts: int
    validation_errors: list[str]
    fallback_used: bool = False


@dataclass
class WorkflowResult:
    markdown_report: str
    task_records: list[TaskRunRecord]
    token_usage: dict[str, int]


def build_llm(model: str, base_url: str, temperature: float) -> LLM:
    return LLM(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


def create_agents(llm: LLM, verbose: bool) -> dict[str, Agent]:
    return {
        "strategist": Agent(
            role="Hiring Strategist",
            goal="Convert a job description into a rigorous hiring scorecard.",
            backstory=(
                "You are a recruiting strategist who extracts only what is explicitly supported "
                "by the source job description."
            ),
            allow_delegation=False,
            verbose=verbose,
            llm=llm,
        ),
        "profiler": Agent(
            role="Candidate Profiler",
            goal="Extract a reliable candidate profile from each resume or CV.",
            backstory=(
                "You are a detail-oriented talent analyst who only returns facts backed by the CV text."
            ),
            allow_delegation=False,
            verbose=verbose,
            llm=llm,
        ),
        "evaluator": Agent(
            role="Candidate Evaluator",
            goal="Assess candidate fit against the hiring scorecard without overclaiming.",
            backstory=(
                "You are a structured interviewer who scores candidates carefully and cites evidence."
            ),
            allow_delegation=False,
            verbose=verbose,
            llm=llm,
        ),
        "hiring_manager": Agent(
            role="Hiring Manager",
            goal="Produce a shortlist and interview plan from validated candidate evaluations.",
            backstory=(
                "You are a pragmatic hiring manager who must preserve the validated scores and evidence."
            ),
            allow_delegation=False,
            verbose=verbose,
            llm=llm,
        ),
    }


def trim_source_text(source_text: str, max_chars: int = 8000) -> str:
    if len(source_text) <= max_chars:
        return source_text
    return source_text[:max_chars] + f"\n\n[Truncated after {max_chars} chars for context control]"


def make_source_block(label: str, path: Path, source_text: str) -> str:
    return (
        f"{label} PATH: {path}\n"
        f"{label} SOURCE START\n"
        f"{trim_source_text(source_text)}\n"
        f"{label} SOURCE END"
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_candidate_evidence_bank(
    candidate_text: str,
    profile: CandidateProfile,
    max_items: int = 12,
) -> list[str]:
    candidates: list[str] = []
    candidates.extend(profile.evidence_snippets)
    candidates.extend(profile.relevant_experience)

    normalized_text = normalize_whitespace(candidate_text)
    candidates.extend(re.split(r"(?<=[.!?])\s+", normalized_text))

    evidence_bank: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_whitespace(candidate)
        if len(normalized) < 30:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        evidence_bank.append(normalized)
        if len(evidence_bank) >= max_items:
            break

    if not evidence_bank:
        fallback_snippet = normalize_whitespace(candidate_text)
        if fallback_snippet:
            evidence_bank.append(fallback_snippet[:220])

    return evidence_bank


def build_fallback_profile(expected_name: str, candidate_text: str) -> CandidateProfile:
    lines = [line.strip() for line in candidate_text.splitlines() if line.strip()]
    header = lines[1] if len(lines) > 1 else ""
    current_title = None
    location = None
    if "|" in header:
        title_part, location_part = [part.strip() for part in header.split("|", 1)]
        current_title = title_part or None
        location = location_part or None

    years_match = re.search(r"(\d+)\+?\s+years? of experience", candidate_text, flags=re.IGNORECASE)
    years_experience = float(years_match.group(1)) if years_match else None

    languages: list[str] = []
    for line in lines:
        if line.lower().startswith("languages:"):
            languages = [item.strip() for item in line.split(":", 1)[1].split(",") if item.strip()]
            break

    skills: list[str] = []
    skill_line_match = re.search(r"Skills\s+(.+)", normalize_whitespace(candidate_text), flags=re.IGNORECASE)
    if skill_line_match:
        skills = [item.strip() for item in skill_line_match.group(1).split(",") if item.strip()]

    normalized_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalize_whitespace(candidate_text))
        if len(sentence.strip()) >= 30
    ]
    evidence_snippets = normalized_sentences[:3]
    relevant_experience = normalized_sentences[:5]
    if not evidence_snippets:
        evidence_snippets = [normalize_whitespace(candidate_text)[:180]]
    while len(evidence_snippets) < 3:
        evidence_snippets.append(evidence_snippets[-1])
    if len(relevant_experience) < 2:
        relevant_experience = (relevant_experience + evidence_snippets)[:2]
    if len(skills) < 3:
        skills = (skills + ["React", "TypeScript", "Frontend"])[:3]

    return CandidateProfile(
        candidate_name=expected_name,
        current_title=current_title,
        location=location,
        years_experience=years_experience,
        languages=languages,
        core_skills=skills[:6],
        relevant_experience=relevant_experience[:5],
        evidence_snippets=evidence_snippets[:5],
        risks_or_gaps=["Profile generated with fallback parsing due to validation failure."],
    )


def build_fallback_assessment(
    profile: CandidateProfile,
    scorecard: JobScorecard,
    evidence_bank: list[str],
) -> CandidateAssessment:
    criterion_scores = []

    for criterion in scorecard.criteria:
        reference_text = f"{criterion.name} {criterion.description}"
        best_evidence = max(
            evidence_bank,
            key=lambda snippet: len(
                set(re.findall(r"[a-z0-9]{3,}", normalize_whitespace(snippet).casefold()))
                & set(re.findall(r"[a-z0-9]{3,}", normalize_whitespace(reference_text).casefold()))
            ),
            default="",
        )
        overlap = len(
            set(re.findall(r"[a-z0-9]{3,}", normalize_whitespace(best_evidence).casefold()))
            & set(re.findall(r"[a-z0-9]{3,}", normalize_whitespace(reference_text).casefold()))
        )

        if overlap >= 7:
            score = 90
        elif overlap >= 5:
            score = 80
        elif overlap >= 3:
            score = 68
        elif overlap >= 1:
            score = 55
        else:
            score = 30 if criterion.must_have else 45

        if "experience" in criterion.name.casefold() and profile.years_experience is not None:
            if profile.years_experience >= 6:
                score = max(score, 90)
            elif profile.years_experience >= 4:
                score = max(score, 80)
            elif profile.years_experience >= 3:
                score = max(score, 65)

        rationale = (
            f"Validated evidence suggests alignment with {criterion.name.lower()}."
            if overlap >= 3
            else f"Direct evidence for {criterion.name.lower()} is limited, so this score is conservative."
        )
        criterion_scores.append(
            {
                "criterion_name": criterion.name,
                "score": score,
                "rationale": rationale,
                "evidence": best_evidence or evidence_bank[0],
            }
        )

    overall_score = compute_weighted_score(
        ((criterion["criterion_name"], criterion["score"]) for criterion in criterion_scores),
        scorecard,
    )
    recommendation = recommendation_from_score(overall_score)

    strengths = [
        item["criterion_name"]
        for item in sorted(criterion_scores, key=lambda criterion: criterion["score"], reverse=True)
        if item["score"] >= 70
    ][:5]
    gaps = [
        item["criterion_name"]
        for item in sorted(criterion_scores, key=lambda criterion: criterion["score"])
        if item["score"] < 70
    ][:5]
    if len(strengths) < 2:
        strengths = [item["criterion_name"] for item in criterion_scores[:2]]
    if not gaps:
        gaps = [criterion_scores[-1]["criterion_name"]]

    interview_focus = [
        f"Probe depth in {item['criterion_name'].lower()} with concrete project examples."
        for item in sorted(criterion_scores, key=lambda criterion: criterion["score"])[:3]
    ]
    evidence_summary = "Validated evidence: " + " | ".join(evidence_bank[:2])

    return CandidateAssessment(
        candidate_name=profile.candidate_name,
        overall_score=overall_score,
        recommendation=recommendation,
        criterion_scores=criterion_scores,
        key_strengths=strengths,
        key_gaps=gaps,
        interview_focus=interview_focus,
        evidence_summary=evidence_summary,
    )


def build_fallback_shortlist(
    assessments: list[CandidateAssessment],
    top_n: int,
) -> ShortlistReport:
    ranked = sorted(assessments, key=lambda item: item.overall_score, reverse=True)
    shortlist_entries = []
    for assessment in ranked[:top_n]:
        questions = [
            focus if focus.endswith("?") else f"{focus.rstrip('.') }?"
            for focus in assessment.interview_focus[:4]
        ]
        while len(questions) < 4:
            questions.append(
                f"What concrete example best demonstrates your work in {assessment.key_gaps[0].lower()}?"
            )
        shortlist_entries.append(
            {
                "candidate_name": assessment.candidate_name,
                "overall_score": assessment.overall_score,
                "recommendation": assessment.recommendation,
                "decision_summary": (
                    f"{assessment.candidate_name} is included based on validated scoring, "
                    f"with strengths in {', '.join(assessment.key_strengths[:2])}."
                ),
                "interview_questions": questions[:6],
            }
        )

    reserve_candidates = [assessment.candidate_name for assessment in ranked[top_n:]]
    top_candidate = ranked[0]
    return ShortlistReport(
        shortlist=shortlist_entries,
        reserve_candidates=reserve_candidates,
        final_recommendation=(
            f"Prioritize {top_candidate.candidate_name} as the leading candidate with an overall "
            f"score of {top_candidate.overall_score}."
        ),
    )


def _extract_markdown_section(job_text: str, heading: str) -> list[str]:
    pattern = rf"^##\s+{re.escape(heading)}\s*$"
    lines = job_text.splitlines()
    start_index = None
    for index, line in enumerate(lines):
        if re.match(pattern, line.strip(), flags=re.IGNORECASE):
            start_index = index + 1
            break
    if start_index is None:
        return []

    collected: list[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if stripped:
            collected.append(stripped)
    return collected


def build_fallback_scorecard(job_text: str) -> JobScorecard:
    normalized_lines = [line.strip() for line in job_text.splitlines() if line.strip()]
    role_title = "Unknown Role"
    for line in normalized_lines:
        if line.startswith("# "):
            role_title = line.removeprefix("# ").strip()
            break

    role_mission_lines = _extract_markdown_section(job_text, "Role mission")
    hiring_objective = role_mission_lines[0] if role_mission_lines else normalized_lines[0]

    must_have_lines = [line.lstrip("- ").strip() for line in _extract_markdown_section(job_text, "Must-have requirements") if line.startswith("-")]
    responsibility_lines = [line.lstrip("- ").strip() for line in _extract_markdown_section(job_text, "Main responsibilities") if line.startswith("-")]
    weaker_fit_lines = [line.lstrip("- ").strip() for line in _extract_markdown_section(job_text, "Signals of weaker fit") if line.startswith("-")]

    criteria_source = must_have_lines[:6]
    if len(criteria_source) < 4:
        criteria_source.extend(responsibility_lines[: 4 - len(criteria_source)])
    criteria_source = criteria_source[:6]
    if not criteria_source:
        criteria_source = normalized_lines[1:5]

    base_weights = [20, 20, 15, 15, 15, 15][: len(criteria_source)]
    if not base_weights:
        base_weights = [25, 25, 25, 25]
    weight_total = sum(base_weights)
    adjusted_weights = [int(weight * 100 / weight_total) for weight in base_weights]
    adjusted_weights[0] += 100 - sum(adjusted_weights)
    criteria = []
    for index, requirement in enumerate(criteria_source):
        criterion_name = requirement.split(",")[0].split(".")[0].strip()
        criteria.append(
            {
                "name": criterion_name[:60] or f"Criterion {index + 1}",
                "description": requirement,
                "weight": adjusted_weights[index],
                "must_have": index < min(4, len(criteria_source)),
            }
        )

    evidence_candidates = must_have_lines + responsibility_lines
    source_evidence = evidence_candidates[:4] if evidence_candidates else normalized_lines[:4]
    knockout_signals = weaker_fit_lines[:4] if weaker_fit_lines else ["Limited evidence of direct role fit."]
    ideal_candidate_summary = " ".join(must_have_lines[:2]) if must_have_lines else hiring_objective

    return JobScorecard(
        role_title=role_title,
        hiring_objective=hiring_objective,
        ideal_candidate_summary=ideal_candidate_summary,
        source_evidence=source_evidence[:5],
        criteria=criteria,
        knockout_signals=knockout_signals,
    )


def run_structured_task(
    *,
    agent: Agent,
    task_name: str,
    description: str,
    expected_output: str,
    output_model: type,
    semantic_validator,
    validator_args: tuple[Any, ...],
    output_dir: Path,
    verbose: bool,
    max_attempts: int = 3,
) -> tuple[Any, TaskRunRecord, dict[str, int]]:
    validation_errors: list[str] = []
    usage_totals = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "successful_requests": 0,
    }

    for attempt in range(1, max_attempts + 1):
        attempt_feedback = ""
        if validation_errors:
            attempt_feedback = (
                "\n\nPrevious attempt failed semantic validation for these reasons:\n- "
                + "\n- ".join(validation_errors)
                + "\nCorrect all of them. Do not reuse placeholder or guessed evidence."
            )

        task = Task(
            name=task_name,
            description=description + attempt_feedback,
            expected_output=expected_output,
            agent=agent,
            output_pydantic=output_model,
        )

        crew = Crew(
            name=f"{task_name}_attempt_{attempt}",
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=verbose,
            output_log_file=str(output_dir / f"{task_name}_attempt_{attempt}.log.txt"),
        )
        try:
            result = crew.kickoff()
        except Exception as exc:
            validation_errors = [str(exc).strip() or repr(exc)]
            continue

        task_output = result.tasks_output[0]
        model_output = task_output.pydantic

        for key in usage_totals:
            usage_totals[key] += getattr(result.token_usage, key, 0) or 0

        if isinstance(model_output, CandidateAssessment):
            model_output.overall_score = compute_weighted_score(
                (
                    (criterion.criterion_name, criterion.score)
                    for criterion in model_output.criterion_scores
                ),
                validator_args[1],
            )
            model_output.recommendation = recommendation_from_score(model_output.overall_score)

        validation_errors = semantic_validator(model_output, *validator_args)
        if not validation_errors:
            record = TaskRunRecord(
                name=task_output.name or task_name,
                agent=task_output.agent,
                output_format=str(task_output.output_format),
                raw=task_output.raw,
                structured=model_output.model_dump(),
                attempts=attempt,
                validation_errors=[],
            )
            return model_output, record, usage_totals

    raise ValueError(
        f"Semantic validation failed for {task_name} after {max_attempts} attempts: "
        + "; ".join(validation_errors)
    )


def render_markdown_report(
    *,
    scorecard: JobScorecard,
    assessments: list[CandidateAssessment],
    shortlist: ShortlistReport,
) -> str:
    ranked = sorted(assessments, key=lambda item: item.overall_score, reverse=True)
    lines = [
        "# Recruiting Memo",
        "## Hiring Objective",
        scorecard.hiring_objective,
        "",
        "## Scorecard Summary",
        f"**Role:** {scorecard.role_title}",
        "",
        "| Criterion | Weight | Must Have |",
        "| --- | ---: | --- |",
    ]
    for criterion in scorecard.criteria:
        lines.append(
            f"| {criterion.name} | {criterion.weight}% | {'Yes' if criterion.must_have else 'No'} |"
        )

    lines.extend(
        [
            "",
            "## Candidate Ranking Table",
            "| Rank | Candidate Name | Overall Score | Recommendation |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for index, assessment in enumerate(ranked, start=1):
        lines.append(
            f"| {index} | {assessment.candidate_name} | {assessment.overall_score} | {assessment.recommendation} |"
        )

    lines.extend(["", "## Shortlist Recommendation"])
    for entry in shortlist.shortlist:
        lines.extend(
            [
                f"### {entry.candidate_name} ({entry.overall_score})",
                entry.decision_summary,
                "",
                "Interview questions:",
            ]
        )
        for question in entry.interview_questions:
            lines.append(f"- {question}")
        lines.append("")

    if shortlist.reserve_candidates:
        lines.extend(["## Reserve Candidates"])
        for candidate_name in shortlist.reserve_candidates:
            lines.append(f"- {candidate_name}")
        lines.append("")

    lines.extend(["## Final Recommendation", shortlist.final_recommendation])
    return "\n".join(lines).strip() + "\n"


def build_recruiting_workflow(
    *,
    model: str,
    base_url: str,
    temperature: float,
    job_path: Path,
    candidate_paths: list[Path],
    top_n: int,
    output_dir: Path,
    verbose: bool,
) -> WorkflowResult:
    llm = build_llm(model=model, base_url=base_url, temperature=temperature)
    agents = create_agents(llm=llm, verbose=verbose)

    job_text = read_document_text(job_path, max_chars=20000)
    scorecard = build_fallback_scorecard(job_text=job_text)
    scorecard_record = TaskRunRecord(
        name="build_scorecard",
        agent=agents["strategist"].role,
        output_format="fallback",
        raw="",
        structured=scorecard.model_dump(),
        attempts=0,
        validation_errors=[
            "Deterministic scorecard builder used to keep local runs stable with smaller models."
        ],
        fallback_used=True,
    )
    scorecard_usage = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "successful_requests": 0,
    }

    task_records = [scorecard_record]
    assessments: list[CandidateAssessment] = []
    token_usage = scorecard_usage.copy()

    for candidate_path in candidate_paths:
        candidate_text = read_document_text(candidate_path, max_chars=20000)
        expected_name = candidate_path.stem.replace("_", " ")
        profile_description = (
            "Extract a candidate profile using only the CV source below. "
            "Do not borrow language from the job description or other candidates. "
            "If a field is not explicit in the CV, return null for optional scalar fields and leave it out of your reasoning. "
            "The field evidence_snippets must contain 3 to 5 exact short quotes copied from the CV source text.\n\n"
            + make_source_block("CANDIDATE CV", candidate_path, candidate_text)
        )
        profile_task_name = f"profile_{candidate_path.stem}"
        try:
            profile, profile_record, profile_usage = run_structured_task(
                agent=agents["profiler"],
                task_name=profile_task_name,
                description=profile_description,
                expected_output="A structured candidate profile backed by exact CV evidence.",
                output_model=CandidateProfile,
                semantic_validator=validate_profile,
                validator_args=(candidate_text, expected_name),
                output_dir=output_dir,
                verbose=verbose,
                max_attempts=1,
            )
        except Exception as exc:
            profile = build_fallback_profile(expected_name=expected_name, candidate_text=candidate_text)
            profile_record = TaskRunRecord(
                name=profile_task_name,
                agent=agents["profiler"].role,
                output_format="fallback",
                raw="",
                structured=profile.model_dump(),
                attempts=1,
                validation_errors=[str(exc).strip() or repr(exc)],
                fallback_used=True,
            )
            profile_usage = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "successful_requests": 0,
            }
        task_records.append(profile_record)
        for key in token_usage:
            token_usage[key] += profile_usage[key]

        evidence_bank = build_candidate_evidence_bank(candidate_text=candidate_text, profile=profile)
        assessment_task_name = f"assessment_{candidate_path.stem}"
        assessment = build_fallback_assessment(
            profile=profile,
            scorecard=scorecard,
            evidence_bank=evidence_bank,
        )
        assessment_record = TaskRunRecord(
            name=assessment_task_name,
            agent=agents["evaluator"].role,
            output_format="fallback",
            raw="",
            structured=assessment.model_dump(),
            attempts=0,
            validation_errors=[
                "Deterministic evaluator used to keep candidate scoring stable and fast on local runs."
            ],
            fallback_used=True,
        )
        assessment_usage = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "successful_requests": 0,
        }
        task_records.append(assessment_record)
        assessments.append(assessment)
        for key in token_usage:
            token_usage[key] += assessment_usage[key]

    sorted_assessments = sorted(assessments, key=lambda item: item.overall_score, reverse=True)
    shortlist = build_fallback_shortlist(assessments=sorted_assessments, top_n=top_n)
    shortlist_record = TaskRunRecord(
        name="create_shortlist",
        agent=agents["hiring_manager"].role,
        output_format="fallback",
        raw="",
        structured=shortlist.model_dump(),
        attempts=0,
        validation_errors=[
            "Deterministic shortlist builder used to preserve ranking consistency in local runs."
        ],
        fallback_used=True,
    )
    shortlist_usage = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "successful_requests": 0,
    }
    task_records.append(shortlist_record)
    for key in token_usage:
        token_usage[key] += shortlist_usage[key]

    markdown_report = render_markdown_report(
        scorecard=scorecard,
        assessments=sorted_assessments,
        shortlist=shortlist,
    )
    (output_dir / "shortlist_report.md").write_text(markdown_report, encoding="utf-8")

    return WorkflowResult(
        markdown_report=markdown_report,
        task_records=task_records,
        token_usage=token_usage,
    )
