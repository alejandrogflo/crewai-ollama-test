from __future__ import annotations

import math
import re
from typing import Iterable

try:
    from .schemas import CandidateAssessment, CandidateProfile, JobScorecard, ShortlistReport
except ImportError:
    from schemas import CandidateAssessment, CandidateProfile, JobScorecard, ShortlistReport


PLACEHOLDER_PATTERNS = (
    "evidence snippet",
    "not provided",
    "n/a",
    "unknown",
    "not available",
)


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_source_text(source_text: str, snippet: str) -> bool:
    source = normalize_text(source_text)
    candidate = normalize_text(snippet)
    return bool(candidate) and candidate in source


def contains_placeholder(text: str) -> bool:
    normalized = normalize_text(text)
    return any(pattern in normalized for pattern in PLACEHOLDER_PATTERNS)


def keyword_overlap(text: str, reference: str) -> int:
    text_tokens = set(re.findall(r"[a-z0-9]{3,}", normalize_text(text)))
    reference_tokens = set(re.findall(r"[a-z0-9]{3,}", normalize_text(reference)))
    return len(text_tokens & reference_tokens)


def validate_scorecard(scorecard: JobScorecard, job_text: str) -> list[str]:
    issues: list[str] = []
    if keyword_overlap(scorecard.role_title, job_text) == 0:
        issues.append("The scorecard role title does not overlap with the job description.")
    if keyword_overlap(scorecard.hiring_objective, job_text) < 3:
        issues.append("The hiring objective is not grounded in the job description text.")
    for snippet in scorecard.source_evidence:
        if contains_placeholder(snippet) or not contains_source_text(job_text, snippet):
            issues.append(f"Job evidence snippet is not present in the source text: {snippet!r}")
    return issues


def validate_profile(
    profile: CandidateProfile,
    candidate_text: str,
    expected_name: str,
) -> list[str]:
    issues: list[str] = []
    if keyword_overlap(profile.candidate_name, expected_name) == 0:
        issues.append("Candidate name does not match the expected candidate document.")
    if keyword_overlap(profile.candidate_name, candidate_text) == 0:
        issues.append("Candidate name is not grounded in the source CV.")

    populated_fields = 0
    if profile.current_title:
        populated_fields += 1
    if profile.location:
        populated_fields += 1
    if profile.years_experience is not None:
        populated_fields += 1
    if profile.languages:
        populated_fields += 1
    if populated_fields < 2:
        issues.append("Too few profile fields were extracted from the CV.")

    for snippet in profile.evidence_snippets:
        if contains_placeholder(snippet) or not contains_source_text(candidate_text, snippet):
            issues.append(f"Candidate evidence snippet is not present in the CV: {snippet!r}")
    return issues


def recommendation_from_score(score: int) -> str:
    if score >= 85:
        return "strong_yes"
    if score >= 70:
        return "yes"
    if score >= 45:
        return "maybe"
    return "no"


def compute_weighted_score(
    criterion_scores: Iterable[tuple[str, int]],
    scorecard: JobScorecard,
) -> int:
    weight_map = {criterion.name: criterion.weight for criterion in scorecard.criteria}
    weighted_total = 0.0
    for criterion_name, score in criterion_scores:
        weighted_total += score * weight_map[criterion_name] / 100
    return int(round(weighted_total))


def validate_assessment(
    assessment: CandidateAssessment,
    allowed_evidence: list[str],
    scorecard: JobScorecard,
) -> list[str]:
    issues: list[str] = []
    expected_names = [criterion.name for criterion in scorecard.criteria]
    actual_names = [criterion.criterion_name for criterion in assessment.criterion_scores]
    if actual_names != expected_names:
        issues.append("Assessment criterion names do not match the scorecard order exactly.")

    for criterion in assessment.criterion_scores:
        if contains_placeholder(criterion.evidence):
            issues.append(f"Criterion evidence uses a placeholder: {criterion.evidence!r}")
        elif not any(
            normalize_text(criterion.evidence) == normalize_text(snippet)
            for snippet in allowed_evidence
        ):
            issues.append(
                f"Criterion evidence is not present in the allowed evidence bank: {criterion.evidence!r}"
            )

    expected_score = compute_weighted_score(
        ((criterion.criterion_name, criterion.score) for criterion in assessment.criterion_scores),
        scorecard,
    )
    if math.fabs(assessment.overall_score - expected_score) > 5:
        issues.append(
            f"Overall score {assessment.overall_score} does not align with weighted criterion score {expected_score}."
        )

    expected_recommendation = recommendation_from_score(expected_score)
    if assessment.recommendation != expected_recommendation:
        issues.append(
            "Recommendation does not align with the weighted score threshold policy."
        )

    if contains_placeholder(assessment.evidence_summary):
        issues.append("Evidence summary contains placeholder content.")
    elif max(
        (
            keyword_overlap(assessment.evidence_summary, snippet)
            for snippet in allowed_evidence
        ),
        default=0,
    ) < 3:
        issues.append("Evidence summary is not grounded in the allowed evidence bank.")

    return issues


def validate_shortlist(
    shortlist: ShortlistReport,
    assessments: list[CandidateAssessment],
    top_n: int,
) -> list[str]:
    issues: list[str] = []
    assessment_map = {assessment.candidate_name: assessment for assessment in assessments}

    if len(shortlist.shortlist) != top_n:
        issues.append(f"Shortlist should contain exactly {top_n} candidates.")

    previous_score: int | None = None
    for entry in shortlist.shortlist:
        assessment = assessment_map.get(entry.candidate_name)
        if assessment is None:
            issues.append(f"Shortlist candidate not found in assessments: {entry.candidate_name}")
            continue
        if entry.overall_score != assessment.overall_score:
            issues.append(
                f"Shortlist score for {entry.candidate_name} does not match the assessment."
            )
        if entry.recommendation != assessment.recommendation:
            issues.append(
                f"Shortlist recommendation for {entry.candidate_name} does not match the assessment."
            )
        if previous_score is not None and entry.overall_score > previous_score:
            issues.append("Shortlist is not sorted by descending score.")
        previous_score = entry.overall_score

    return issues
