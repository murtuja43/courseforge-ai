"""Live AI course generation through the Anthropic API, with an offline fallback.

The model receives the selected pages tagged with their physical PDF page numbers and
must return lessons as JSON matching the existing schema. Every response is checked by
parse_course() and check_citations(); only the validator can mark a citation verified.
Rejected output gets a limited number of corrective retries that list the exact
problems. If citations still fail, the result is returned as an explicitly unverified
draft. A verified citation shows that the quote appears on the cited page; it does not
show that the AI explanation around it is correct.

Configuration comes from environment variables (or a local .env file):
LLM_PROVIDER, ANTHROPIC_API_KEY, COURSEFORGE_MODEL and optionally COURSEFORGE_EFFORT.
The API key is never logged, printed or included in error messages.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

import anthropic
from pydantic import BaseModel

from courseforge.generator import (
    DemoResult, InsufficientSourceError, check_selection, generate_demo_course,
    insufficient_source_message,
)
from courseforge.pdf_extract import Selection
from courseforge.schema import LESSON_COUNT, MAX_QUOTE_CHARS, MIN_QUOTE_CHARS, Course, Lesson
from courseforge.validate import TRACEABILITY_NOTE, CitationReport, check_citations, parse_course

log = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("anthropic",)
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
MAX_ATTEMPTS = 3                 # one generation plus at most two corrective retries
MAX_OUTPUT_TOKENS = 32_000
MAX_SOURCE_CHARS = 150_000       # roughly 40k tokens of chapter text
MAX_PROBLEMS_IN_FEEDBACK = 30
REQUEST_TIMEOUT_SECONDS = 600.0
SDK_MAX_RETRIES = 2              # the SDK retries 408/409/429/5xx and connection errors

TEACHER_CHECK = "Teacher check:"

SYSTEM_PROMPT = f"""You help a teacher turn textbook pages into a {LESSON_COUNT}-lesson \
mini-course. The teacher reviews everything before students see it.

Teach, do not copy. Explain each concept in your own clear, student-friendly words, building \
from simpler ideas to harder ones: say what the method is for, how it works step by step, \
and when to use it. Do not string together sentences copied from the textbook; quotations \
belong only in the source references. Every fact must come from the source pages provided. \
Do not add facts, definitions, numbers, names or results that the source does not support.

Structure:
- Exactly {LESSON_COUNT} lessons, numbered 1 to {LESSON_COUNT}, with distinct titles, in \
the order of the source.
- When a list of textbook sections is given, organize the lessons around it: a lesson \
covers one section, or one coherent part of a longer section, and never mixes unrelated \
sections. Give longer or harder sections more lessons. Text before the first listed \
heading on a page may belong to the end of an earlier section; use it only as context.
- Lesson titles name the concept taught (for example "Solving complex systems with real \
arithmetic"), not page numbers.
- Each lesson has 2 to 4 specific, measurable learning objectives; a teaching explanation \
of a few paragraphs; at least one example; at least one practice activity with a \
suggested answer; and 2 or 3 multiple-choice quiz questions.
- Examples: when the source works through an example, present it step by step using the \
source's own numbers. If the example is incomplete on these pages, or you work out steps \
the source does not show, say so in the example, beginning that sentence with \
"{TEACHER_CHECK}", so the teacher can verify it.
- Quiz questions have 4 options with ids A to D and exactly one correct option. Wrong \
options must be plausible but clearly wrong according to the source. Calculation questions \
use values from the source; check the arithmetic so exactly one option is correct. The \
explanation says why the correct answer is right.

Mathematics:
- Equations are essential; teach them. Present clearly extracted equations with their \
equation numbers and explain what each term means.
- The page text was extracted from a PDF, and equations may be garbled: fractions split \
across lines, matrices broken into single symbols, or symbols such as summation signs \
missing. Never present your own reconstruction of a garbled equation as if it were the \
textbook's equation. Refer to it by its number and page (for example "Eq. (9.36) on page \
288"), explain it using the reliable surrounding prose, and add a sentence beginning \
"{TEACHER_CHECK}" saying that the equation must be checked against the printed page.
- If you write a mathematical expression yourself, for example to explain a step, label \
it as an explanation rather than a quotation from the textbook.

Source references:
- Every lesson, example, practice activity and quiz question has at least one source \
reference.
- "page" is the physical PDF page number from the page tag, for example \
<page number="286" printed="267"> means page 286. Never use the printed page number. Cite \
only pages you were given.
- "quote" is copied character for character from that one page: {MIN_QUOTE_CHARS} to \
{MAX_QUOTE_CHARS} characters, one continuous passage, with no ellipses and no added or \
changed words. It never spans two pages. Prefer prose sentences, because garbled equation \
text is not reliable to quote. Never invent a quotation.
- Choose a quote that directly supports the content it is attached to.
- You cannot verify citations. Never state that a quotation or source has been verified \
or checked; CourseForge checks every quotation after you respond."""


class LLMError(Exception):
    """AI generation failed. The message is safe to show to a teacher."""


class LLMConfigError(LLMError):
    """AI generation is not configured, or the configured model is not available."""


class LLMAuthError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    def __init__(self, message: str, retry_after: str | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


class LLMRequestError(LLMError):
    pass


class LLMServiceError(LLMError):
    pass


class LLMOutputError(LLMError):
    """The model did not produce an acceptable course."""

    def __init__(self, message: str, problems: list[str] | None = None):
        super().__init__(message)
        self.problems = problems or []


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str = field(repr=False)
    effort: str | None = None


class CourseDraft(BaseModel):
    """What the model returns; metadata is added by CourseForge, not by the model."""
    lessons: list[Lesson]


@dataclass
class LLMResult:
    course: Course
    citations: CitationReport
    attempts: int
    problems: list[str] = field(default_factory=list)   # unresolved problems, if any
    notices: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.citations.all_verified and not self.problems

    @property
    def status_message(self) -> str:
        model = self.course.metadata.model
        if self.verified:
            check = f"All {len(self.citations.checks)} citations were found on their cited pages."
        else:
            check = (f"UNVERIFIED DRAFT: {len(self.problems)} problem(s) remain after "
                     f"{self.attempts} attempt(s); see the citation report.")
        return (f"Generated by the AI model {model}. {check} {TRACEABILITY_NOTE} "
                "This is a draft and must be reviewed by a teacher.")


# ---------------------------------------------------------------- configuration

def load_config(env: Mapping[str, str] | None = None, *, use_dotenv: bool = True) -> LLMConfig:
    """Read AI settings from the environment. Raises LLMConfigError with a clear reason."""
    if env is None:
        if use_dotenv:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        env = os.environ
    provider = env.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        raise LLMConfigError("AI generation is off because LLM_PROVIDER is not set.")
    if provider not in SUPPORTED_PROVIDERS:
        raise LLMConfigError(f"LLM_PROVIDER '{provider}' is not supported; use "
                             + " or ".join(SUPPORTED_PROVIDERS) + ".")
    api_key = env.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigError("AI generation is off because ANTHROPIC_API_KEY is not set.")
    model = env.get("COURSEFORGE_MODEL", "").strip()
    if not model:
        raise LLMConfigError("AI generation is off because COURSEFORGE_MODEL is not set. "
                             "Set it to a model your API key can access.")
    effort = env.get("COURSEFORGE_EFFORT", "").strip().lower() or None
    if effort and effort not in EFFORT_LEVELS:
        raise LLMConfigError(f"COURSEFORGE_EFFORT '{effort}' is not valid; use one of "
                             + ", ".join(EFFORT_LEVELS) + ", or leave it empty.")
    return LLMConfig(provider=provider, model=model, api_key=api_key, effort=effort)


def make_client(config: LLMConfig) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=config.api_key, timeout=REQUEST_TIMEOUT_SECONDS,
                               max_retries=SDK_MAX_RETRIES)


def verify_model(client, model: str) -> None:
    """Check that the configured model exists for this key (no tokens are used)."""
    try:
        client.models.retrieve(model)
    except anthropic.APIError as exc:
        raise _translate(exc, model) from exc


def _translate(exc: anthropic.APIError, model: str) -> LLMError:
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return LLMAuthError("The AI service rejected the API key. Check ANTHROPIC_API_KEY.")
    if isinstance(exc, anthropic.NotFoundError):
        return LLMConfigError(f"The model '{model}' was not found or is not available to this "
                              "API key. Check COURSEFORGE_MODEL.")
    if isinstance(exc, anthropic.RateLimitError):
        retry_after = exc.response.headers.get("retry-after")
        wait = f" Try again in about {retry_after} seconds." if retry_after else " Try again later."
        return LLMRateLimitError("The AI service rate limit was reached." + wait, retry_after)
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError("The AI service did not respond in time. Try again, or select fewer pages.")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMConnectionError("Could not connect to the AI service. Check the internet connection.")
    if isinstance(exc, anthropic.BadRequestError):
        return LLMRequestError(f"The AI service rejected the request: {exc.message}")
    if isinstance(exc, anthropic.APIStatusError):
        return LLMServiceError(f"The AI service returned an error (HTTP {exc.status_code}). Try again later.")
    return LLMError("The AI request failed.")


# ---------------------------------------------------------------- generation

@lru_cache(maxsize=1)
def _output_schema() -> dict:
    return anthropic.transform_schema(CourseDraft)


def build_source_text(selection: Selection) -> str:
    """The usable selected pages, each tagged with its physical PDF page number.

    The printed page label is added as a separate attribute when it differs, so the
    model can see both but is told to cite only the physical number.
    """
    def tag(p) -> str:
        printed = f' printed="{p.label}"' if p.label and p.label != str(p.page_number) else ""
        return f'<page number="{p.page_number}"{printed}>\n{p.text}\n</page>'
    return "\n\n".join(tag(p) for p in selection.text_pages)


def build_section_context(selection: Selection) -> str:
    """The PDF's own table-of-contents sections that overlap the selection."""
    if not selection.sections:
        return "The PDF has no table of contents for these pages, so no section list is available."
    lines = ["Textbook sections overlapping these pages, from the PDF's table of contents "
             "(physical PDF page numbers):"]
    for s in selection.sections:
        indent = "  " * (s.level - 1)
        if s.start_page < selection.start_page:
            where = f"began on page {s.start_page}, before the selected pages; continues here"
        else:
            where = f"starts on page {s.start_page}"
        if s.end_page > selection.end_page:
            where += "; continues after the selected pages"
        elif s.end_shared and s.end_page <= selection.end_page:
            where += f"; ends partway down page {s.end_page}"
        lines.append(f"{indent}- {s.title} ({where})")
    return "\n".join(lines)


def teacher_check_locations(course: Course) -> list[str]:
    """Where the model flagged something for the teacher to check."""
    found = []
    marker = TEACHER_CHECK.casefold()
    for lesson in course.lessons:
        where = f"Lesson {lesson.number}"
        texts = [(where, lesson.explanation)]
        texts += [(f"{where}, example {i}", e.explanation) for i, e in enumerate(lesson.examples, 1)]
        texts += [(f"{where}, practice activity {i}", f"{a.prompt} {a.suggested_answer}")
                  for i, a in enumerate(lesson.practice, 1)]
        texts += [(f"{where}, quiz question {i}", f"{q.question} {q.explanation}")
                  for i, q in enumerate(lesson.quiz, 1)]
        found += [loc for loc, text in texts if marker in text.casefold()]
    return found


def _notices(selection: Selection, course: Course) -> list[str]:
    notices = list(selection.warnings)
    flagged = teacher_check_locations(course)
    if flagged:
        notices.append(f"The AI flagged {len(flagged)} item(s) for teacher checking (marked "
                       f"\"{TEACHER_CHECK}\"): " + "; ".join(flagged) + ".")
    return notices


def generate_llm_course(selection: Selection, source_filename: str, *,
                        config: LLMConfig, client=None, chapter_label: str | None = None,
                        document_page_count: int | None = None, created_at: datetime | None = None,
                        max_attempts: int = MAX_ATTEMPTS) -> LLMResult:
    """Generate a course with the AI model, validating schema and citations after each attempt."""
    check_selection(selection, document_page_count)
    if not selection.has_usable_text:
        raise InsufficientSourceError(insufficient_source_message(selection, 0))
    source = build_source_text(selection)
    if len(source) > MAX_SOURCE_CHARS:
        raise LLMRequestError(f"The selected pages contain {len(source):,} characters of text; the "
                              f"limit is {MAX_SOURCE_CHARS:,}. Select a shorter chapter or page range.")

    client = client or make_client(config)
    verify_model(client, config.model)

    label = chapter_label or selection.label
    metadata = {"title": label, "source_filename": source_filename, "chapter_label": label,
                "start_page": selection.start_page, "end_page": selection.end_page,
                "generation_mode": "llm", "model": config.model, "review_status": "draft"}
    if created_at is not None:
        metadata["created_at"] = created_at

    messages: list[dict] = [{"role": "user", "content": [
        {"type": "text", "cache_control": {"type": "ephemeral"},  # reused by corrective retries
         "text": (f"Selection: {label}\nPhysical PDF pages {selection.start_page}-{selection.end_page}. "
                  f"Pages without extractable text are left out.\n\n"
                  f"{build_section_context(selection)}\n\n{source}")},
        {"type": "text", "text": f"Create the {LESSON_COUNT}-lesson course from these pages."},
    ]}]
    usage = {"input_tokens": 0, "output_tokens": 0}
    best: tuple[Course, CitationReport, list[str]] | None = None
    problems: list[str] = []

    for attempt in range(1, max_attempts + 1):
        text = _request(client, config, messages, usage)
        course, report, problems = _evaluate(text, metadata, selection, document_page_count)
        log.info("AI course attempt %d of %d: %d problem(s)", attempt, max_attempts, len(problems))
        if course is not None and not problems:
            return LLMResult(course, report, attempt, notices=_notices(selection, course), usage=usage)
        if course is not None:
            best = (course, report, problems)
        if attempt < max_attempts:
            if text.strip():
                messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": _feedback(problems)})

    if best is not None:
        course, report, remaining = best
        return LLMResult(course, report, max_attempts, problems=remaining,
                         notices=_notices(selection, course), usage=usage)
    raise LLMOutputError(f"The AI model did not return a valid course after {max_attempts} "
                         "attempt(s). Try again, or use demo mode.", problems)


def _request(client, config: LLMConfig, messages: list[dict], usage: dict[str, int]) -> str:
    output_config: dict = {"format": {"type": "json_schema", "schema": _output_schema()}}
    if config.effort:
        output_config["effort"] = config.effort
    try:
        with client.messages.stream(model=config.model, max_tokens=MAX_OUTPUT_TOKENS,
                                    system=SYSTEM_PROMPT, messages=list(messages),
                                    output_config=output_config) as stream:
            message = stream.get_final_message()
    except anthropic.APIError as exc:
        raise _translate(exc, config.model) from exc

    for key in usage:
        usage[key] += getattr(message.usage, key, 0) or 0
    if message.stop_reason == "refusal":
        raise LLMOutputError("The AI model declined to process these pages. Try demo mode.")
    if message.stop_reason == "max_tokens":
        raise LLMOutputError("The AI response was too long and was cut off. Select fewer pages.")
    return "".join(block.text for block in message.content if block.type == "text")


def _evaluate(text: str, metadata: dict, selection: Selection,
              document_page_count: int | None) -> tuple[Course | None, CitationReport | None, list[str]]:
    if not text.strip():
        return None, None, ["The response was empty."]
    try:
        draft = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, None, [f"The response was not valid JSON: {exc.msg}."]
    if not isinstance(draft, dict) or set(draft) != {"lessons"}:
        return None, None, ['The response must be a JSON object with only a "lessons" list.']

    course, errors = parse_course({"metadata": metadata, "lessons": draft["lessons"]})
    if course is None:
        return None, None, errors
    report = check_citations(course, selection, document_page_count)
    problems = [f"{c.location}: {c.message}" for c in report.problems]
    problems += [f"Course: {e}" for e in report.course_errors]
    problems += [f"{item}: has no source reference; every item needs one." for item in report.uncited]
    return course, report, problems


def _feedback(problems: list[str]) -> str:
    shown = problems[:MAX_PROBLEMS_IN_FEEDBACK]
    more = len(problems) - len(shown)
    lines = "\n".join(f"- {p}" for p in shown) + (f"\n- ...and {more} more." if more > 0 else "")
    return ("Your course could not be accepted because of these problems:\n" + lines + "\n\n"
            "Return the complete corrected course as JSON. For each citation problem, copy the "
            "quote exactly from the cited page, or cite the page that really contains it.")


# ---------------------------------------------------------------- choosing a generator

def generate_course(selection: Selection, source_filename: str, *, mode: str = "auto",
                    config: LLMConfig | None = None, client=None, **kwargs) -> DemoResult | LLMResult:
    """Generate with AI when configured, otherwise with the offline demo generator.

    mode "auto" falls back to demo mode only when AI is not configured; runtime errors
    such as rate limits are raised so the caller can decide (for example, offer demo mode).
    """
    if mode == "demo":
        return generate_demo_course(selection, source_filename, **kwargs)
    if mode not in ("auto", "llm"):
        raise ValueError(f"Unknown generation mode: {mode!r}")
    try:
        config = config or load_config()
    except LLMConfigError as exc:
        if mode == "llm":
            raise
        result = generate_demo_course(selection, source_filename, **kwargs)
        result.notices.insert(0, f"{exc} The offline demo generator was used instead.")
        return result
    return generate_llm_course(selection, source_filename, config=config, client=client, **kwargs)
