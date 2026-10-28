"""Typed data model for a generated mini-course.

The schema checks structure only: required fields, non-empty text, valid quiz options
and answers, and a five-lesson course. Whether each source reference really points to
supporting text on its page is checked separately in courseforge.validate.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

LESSON_COUNT = 5
MIN_QUOTE_CHARS = 15
MAX_QUOTE_CHARS = 300

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OptionId = Annotated[str, StringConstraints(strip_whitespace=True, to_upper=True, pattern=r"^[A-Fa-f]$")]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceReference(_Model):
    page: int = Field(ge=1, description="One-based physical PDF page number")
    quote: Annotated[str, StringConstraints(strip_whitespace=True, min_length=MIN_QUOTE_CHARS,
                                            max_length=MAX_QUOTE_CHARS)] = Field(
        description="Short passage copied exactly from the extracted text of that page")


Sources = Annotated[list[SourceReference], Field(min_length=1)]


class LearningObjective(_Model):
    text: Text


class Example(_Model):
    title: Text
    explanation: Text
    sources: list[SourceReference] = []


class PracticeActivity(_Model):
    prompt: Text
    suggested_answer: Text
    sources: list[SourceReference] = []


class QuizOption(_Model):
    id: OptionId
    text: Text


class QuizQuestion(_Model):
    question: Text
    options: Annotated[list[QuizOption], Field(min_length=2, max_length=6)]
    correct_option: OptionId
    explanation: Text
    sources: Sources

    @model_validator(mode="after")
    def _check_options(self) -> QuizQuestion:
        ids = [o.id for o in self.options]
        if len(set(ids)) != len(ids):
            raise ValueError(f"option ids must be unique, got {ids}")
        texts = [o.text.casefold() for o in self.options]
        if len(set(texts)) != len(texts):
            raise ValueError("option texts must be different from each other")
        if self.correct_option not in ids:
            raise ValueError(f"correct_option {self.correct_option!r} is not one of the options {ids}")
        return self

    @property
    def correct_answer(self) -> QuizOption:
        return next(o for o in self.options if o.id == self.correct_option)


class Lesson(_Model):
    number: int = Field(ge=1, le=LESSON_COUNT)
    title: Text
    objectives: Annotated[list[LearningObjective], Field(min_length=1)]
    explanation: Text
    examples: Annotated[list[Example], Field(min_length=1)]
    practice: Annotated[list[PracticeActivity], Field(min_length=1)]
    quiz: Annotated[list[QuizQuestion], Field(min_length=1)]
    sources: Sources


class CourseMetadata(_Model):
    title: Text
    source_filename: Text
    chapter_label: Text
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    generation_mode: Literal["demo", "llm"]
    model: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    review_status: Literal["draft", "approved"] = "draft"

    @model_validator(mode="after")
    def _check_range(self) -> CourseMetadata:
        if self.start_page > self.end_page:
            raise ValueError("start_page must not be after end_page")
        return self


class Course(_Model):
    metadata: CourseMetadata
    lessons: Annotated[list[Lesson], Field(min_length=LESSON_COUNT, max_length=LESSON_COUNT)]

    @model_validator(mode="after")
    def _check_lessons(self) -> Course:
        numbers = [lesson.number for lesson in self.lessons]
        if numbers != list(range(1, LESSON_COUNT + 1)):
            raise ValueError(f"lessons must be numbered 1 to {LESSON_COUNT} in order, got {numbers}")
        titles = [lesson.title.casefold() for lesson in self.lessons]
        if len(set(titles)) != len(titles):
            raise ValueError("lesson titles must be different from each other")
        return self

    def iter_sources(self) -> Iterator[tuple[str, SourceReference]]:
        """Every source reference with a readable location, e.g. "Lesson 2, quiz question 1"."""
        for lesson in self.lessons:
            where = f"Lesson {lesson.number}"
            for ref in lesson.sources:
                yield where, ref
            for i, ex in enumerate(lesson.examples, 1):
                for ref in ex.sources:
                    yield f"{where}, example {i}", ref
            for i, act in enumerate(lesson.practice, 1):
                for ref in act.sources:
                    yield f"{where}, practice activity {i}", ref
            for i, q in enumerate(lesson.quiz, 1):
                for ref in q.sources:
                    yield f"{where}, quiz question {i}", ref

    def uncited_items(self) -> list[str]:
        """Examples and practice activities without their own source references."""
        missing = []
        for lesson in self.lessons:
            where = f"Lesson {lesson.number}"
            missing += [f"{where}, example {i}" for i, ex in enumerate(lesson.examples, 1) if not ex.sources]
            missing += [f"{where}, practice activity {i}"
                        for i, act in enumerate(lesson.practice, 1) if not act.sources]
        return missing
