from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ScoringCriterion(BaseModel):
    name: str = Field(description="Short criterion name.")
    description: str = Field(description="What good performance looks like.")
    weight: int = Field(ge=5, le=50, description="Weight percentage for this criterion.")
    must_have: bool = Field(
        default=False,
        description="Whether failing this criterion should seriously hurt the recommendation.",
    )


class JobScorecard(BaseModel):
    role_title: str
    hiring_objective: str
    ideal_candidate_summary: str
    source_evidence: list[str] = Field(
        min_length=3,
        max_length=5,
        description="Exact short quotes copied from the job description.",
    )
    criteria: list[ScoringCriterion] = Field(min_length=4, max_length=8)
    knockout_signals: list[str] = Field(
        min_length=1,
        description="Signals that should lower confidence or disqualify a candidate.",
    )

    @field_validator("role_title", "hiring_objective", "ideal_candidate_summary")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be empty.")
        return value

    @field_validator("source_evidence", "knockout_signals", mode="before")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not cleaned:
            raise ValueError("List cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_total_weight(self) -> "JobScorecard":
        total_weight = sum(item.weight for item in self.criteria)
        if total_weight != 100:
            raise ValueError(f"Criterion weights must sum to 100, got {total_weight}.")
        return self


class CandidateProfile(BaseModel):
    candidate_name: str
    current_title: str | None = None
    location: str | None = None
    years_experience: float | None = Field(default=None, ge=0, le=50)
    languages: list[str] = Field(default_factory=list)
    core_skills: list[str] = Field(min_length=3)
    relevant_experience: list[str] = Field(
        min_length=2,
        description="Bullets about the candidate's most relevant background.",
    )
    evidence_snippets: list[str] = Field(
        min_length=3,
        max_length=5,
        description="Exact short quotes copied from the source document.",
    )
    risks_or_gaps: list[str] = Field(default_factory=list)

    @field_validator("candidate_name")
    @classmethod
    def validate_candidate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Candidate name cannot be empty.")
        return value

    @field_validator("current_title", "location")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator(
        "languages",
        "core_skills",
        "relevant_experience",
        "evidence_snippets",
        "risks_or_gaps",
        mode="before",
    )
    @classmethod
    def normalize_string_lists(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]


class CriterionScore(BaseModel):
    criterion_name: str
    score: int = Field(ge=0, le=100)
    rationale: str
    evidence: str

    @field_validator("criterion_name", "rationale", "evidence")
    @classmethod
    def validate_criterion_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Criterion text fields cannot be empty.")
        return value


class CandidateAssessment(BaseModel):
    candidate_name: str
    overall_score: int = Field(ge=0, le=100)
    recommendation: Literal["strong_yes", "yes", "maybe", "no"]
    criterion_scores: list[CriterionScore] = Field(min_length=4, max_length=8)
    key_strengths: list[str] = Field(min_length=2, max_length=5)
    key_gaps: list[str] = Field(min_length=1, max_length=5)
    interview_focus: list[str] = Field(
        min_length=3,
        max_length=6,
        description="Topics to probe if the candidate progresses.",
    )
    evidence_summary: str

    @field_validator("candidate_name", "evidence_summary")
    @classmethod
    def validate_assessment_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Assessment text fields cannot be empty.")
        return value

    @field_validator("key_strengths", "key_gaps", "interview_focus", mode="before")
    @classmethod
    def normalize_assessment_lists(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not cleaned:
            raise ValueError("Assessment lists cannot be empty.")
        return cleaned


class ShortlistEntry(BaseModel):
    candidate_name: str
    overall_score: int = Field(ge=0, le=100)
    recommendation: Literal["strong_yes", "yes", "maybe", "no"]
    decision_summary: str
    interview_questions: list[str] = Field(min_length=4, max_length=6)

    @field_validator("candidate_name", "decision_summary")
    @classmethod
    def validate_shortlist_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Shortlist text fields cannot be empty.")
        return value

    @field_validator("interview_questions", mode="before")
    @classmethod
    def normalize_questions(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if len(cleaned) < 4:
            raise ValueError("Each shortlisted candidate needs at least 4 questions.")
        return cleaned


class ShortlistReport(BaseModel):
    shortlist: list[ShortlistEntry] = Field(min_length=1)
    reserve_candidates: list[str] = Field(default_factory=list)
    final_recommendation: str

    @field_validator("reserve_candidates", mode="before")
    @classmethod
    def normalize_reserve_candidates(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    @field_validator("final_recommendation")
    @classmethod
    def validate_final_recommendation(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Final recommendation cannot be empty.")
        return value
