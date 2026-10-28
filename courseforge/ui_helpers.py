"""Interface logic for the Streamlit app, kept free of Streamlit so it can be tested.

Edits: every editable field of the course has a stable widget key built from the course
id and the field's path. The app seeds those keys once and reads them back on every run,
so switching lessons never discards an edit.

Approval: a course counts as approved only while its content matches the fingerprint
taken when the teacher approved it. Any later edit returns it to draft.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field

from courseforge.generator import DemoGenerationError, InsufficientSourceError, InvalidSelectionError
from courseforge.llm import (
    LLMAuthError, LLMConfigError, LLMError, LLMOutputError, LLMRateLimitError, generate_course,
    load_config,
)
from courseforge.pdf_extract import Chapter, ExtractedDocument, Selection
from courseforge.schema import Course
from courseforge.validate import (
    TRACEABILITY_NOTE, CitationCheck, CitationReport, CitationStatus, check_citations, parse_course,
)

MODES = {
    "auto": "Auto: use AI when it is configured, otherwise the offline demo",
    "llm": "AI mode: requires a configured API key and model",
    "demo": "Demo mode: offline, deterministic, no AI",
}

CITATION_LABELS = {
    CitationStatus.VERIFIED: "Verified",
    CitationStatus.QUOTE_NOT_FOUND: "Not verified: quote not found on the page",
    CitationStatus.OUTSIDE_SELECTION: "Not verified: outside the selected chapter",
    CitationStatus.PAGE_NOT_IN_DOCUMENT: "Not verified: page does not exist",
    CitationStatus.PAGE_HAS_NO_TEXT: "Cannot check: page has no usable text",
}
MISSING_LABEL = "Missing: no source reference"

_NOT_EDITABLE = {"number", "id"}


# ---------------------------------------------------------------- AI configuration

def ai_status() -> tuple[bool, str]:
    """Whether AI generation is configured, with a message that never includes the key."""
    try:
        config = load_config()
    except LLMConfigError as exc:
        return False, str(exc)
    return True, f"AI is configured with the model {config.model} (checked with the provider when you generate)."


# ---------------------------------------------------------------- chapters

def chapter_label(chapter: Chapter) -> str:
    pages = (f"page {chapter.start_page}" if chapter.start_page == chapter.end_page
             else f"pages {chapter.start_page}-{chapter.end_page}")
    indent = " " * (chapter.level - 1)
    return f"{indent}{chapter.title} ({pages})"


# ---------------------------------------------------------------- generation

@dataclass
class GenerationOutcome:
    result: object | None = None        # DemoResult or LLMResult
    title: str = ""
    message: str = ""
    details: list[str] = field(default_factory=list)
    offer_demo: bool = False

    @property
    def ok(self) -> bool:
        return self.result is not None


def run_generation(selection: Selection, doc: ExtractedDocument, mode: str, *,
                   config=None, client=None) -> GenerationOutcome:
    """Run the chosen generator and turn every expected failure into a displayable outcome.

    A failed AI request is never replaced by a demo course here; the outcome only offers
    demo mode so the teacher can choose it.
    """
    try:
        result = generate_course(selection, doc.filename, mode=mode, config=config, client=client,
                                 document_page_count=doc.page_count)
    except LLMConfigError as exc:
        return GenerationOutcome(title="AI is not available", message=str(exc), offer_demo=True)
    except LLMAuthError as exc:
        return GenerationOutcome(title="The API key was rejected", message=str(exc), offer_demo=True)
    except LLMRateLimitError as exc:
        return GenerationOutcome(title="AI rate limit reached", message=str(exc), offer_demo=True)
    except LLMOutputError as exc:
        return GenerationOutcome(title="The AI output could not be accepted", message=str(exc),
                                 details=exc.problems, offer_demo=True)
    except LLMError as exc:
        return GenerationOutcome(title="The AI request failed", message=str(exc), offer_demo=True)
    except InsufficientSourceError as exc:
        return GenerationOutcome(title="Not enough usable text", message=str(exc))
    except InvalidSelectionError as exc:
        return GenerationOutcome(title="Invalid page selection", message=str(exc))
    except DemoGenerationError as exc:
        return GenerationOutcome(title="The demo course could not be built", message=str(exc))
    return GenerationOutcome(result=result)


def generation_info(result) -> dict:
    """The parts of a generation result the app keeps next to the editable course."""
    meta = result.course.metadata
    return {"mode": meta.generation_mode, "model": meta.model,
            "attempts": getattr(result, "attempts", None), "usage": getattr(result, "usage", None),
            "notices": list(result.notices)}


# ---------------------------------------------------------------- editing

def widget_key(course_id: int, path: tuple) -> str:
    return f"edit|{course_id}|" + ".".join(str(p) for p in path)


def editable_fields(course_data: dict) -> list[tuple[tuple, object]]:
    """(path, value) for every editable text or page-number field in the lessons."""
    fields: list[tuple[tuple, object]] = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key not in _NOT_EDITABLE:
                    walk(value, path + (key,))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, path + (i,))
        elif isinstance(node, (str, int)) and not isinstance(node, bool):
            fields.append((path, node))

    walk(course_data["lessons"], ("lessons",))
    return fields


def seed_edit_state(state: MutableMapping, course_id: int, course_data: dict) -> None:
    """Give every edit widget its starting value, without overwriting existing edits."""
    for path, value in editable_fields(course_data):
        state.setdefault(widget_key(course_id, path), value)


def collect_edits(state: MutableMapping, course_id: int, course_data: dict) -> dict:
    """A copy of the course with the current widget values applied."""
    edited = copy.deepcopy(course_data)
    for path, _ in editable_fields(course_data):
        key = widget_key(course_id, path)
        if key in state:
            target = edited
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = state[key]
    return edited


def source_paths(course_data: dict) -> Iterator[tuple]:
    """Paths of every source list entry, in the same order as Course.iter_sources()."""
    for li, lesson in enumerate(course_data["lessons"]):
        base = ("lessons", li)
        for si in range(len(lesson["sources"])):
            yield base + ("sources", si)
        for section in ("examples", "practice", "quiz"):
            for ii, item in enumerate(lesson[section]):
                for si in range(len(item["sources"])):
                    yield base + (section, ii, "sources", si)


@dataclass
class Evaluation:
    course: Course | None
    report: CitationReport | None
    errors: list[str]
    checks_by_path: dict[tuple, CitationCheck]


def evaluate_course(course_data: dict, selection: Selection, page_count: int) -> Evaluation:
    """Validate the edited course and re-check every citation against the selected pages."""
    course, errors = parse_course(course_data)
    if course is None:
        return Evaluation(None, None, errors, {})
    report = check_citations(course, selection, page_count)
    checks = dict(zip(source_paths(course_data), report.checks))
    return Evaluation(course, report, [], checks)


# ---------------------------------------------------------------- approval

def fingerprint(course: Course) -> str:
    content = json.dumps(course.model_dump(mode="json")["lessons"], sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def with_review_status(course: Course, approved_fingerprint: str | None) -> Course:
    """The course marked approved only if it still matches the approved content."""
    status = "approved" if approved_fingerprint and approved_fingerprint == fingerprint(course) else "draft"
    return course.model_copy(update={"metadata": course.metadata.model_copy(update={"review_status": status})})


# ---------------------------------------------------------------- citation reporting

def citation_label(check: CitationCheck) -> str:
    return CITATION_LABELS[check.status]


def citation_rows(report: CitationReport) -> list[dict]:
    rows = [{"Location": c.location, "PDF page": c.page, "Status": citation_label(c),
             "Quote": c.quote, "Detail": c.message} for c in report.checks]
    rows += [{"Location": item, "PDF page": None, "Status": MISSING_LABEL, "Quote": "",
              "Detail": "This item has no source reference."} for item in report.uncited]
    return rows


def citation_counts(report: CitationReport) -> dict[str, int]:
    checks = report.checks
    return {
        "verified": sum(c.verified for c in checks),
        "not_verified": sum(c.status in (CitationStatus.QUOTE_NOT_FOUND, CitationStatus.OUTSIDE_SELECTION,
                                         CitationStatus.PAGE_NOT_IN_DOCUMENT) for c in checks),
        "cannot_check": sum(c.status == CitationStatus.PAGE_HAS_NO_TEXT for c in checks),
        "missing": len(report.uncited),
    }


# ---------------------------------------------------------------- export

def mode_label(course: Course) -> str:
    meta = course.metadata
    if meta.generation_mode == "llm":
        return f"AI-generated (model: {meta.model})"
    return "Offline demo (assembled from the source text by fixed rules; no AI used)"


def course_to_json(course: Course, report: CitationReport) -> str:
    data = {
        "course": course.model_dump(mode="json"),
        "citation_check": {
            "summary": citation_counts(report),
            "all_verified": report.all_verified,
            "results": [{"location": c.location, "page": c.page, "quote": c.quote,
                         "status": c.status.value, "detail": c.message} for c in report.checks],
            "uncited_items": report.uncited,
            "course_errors": report.course_errors,
            "note": TRACEABILITY_NOTE,
        },
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def course_to_markdown(course: Course, report: CitationReport) -> str:
    meta = course.metadata
    status_by_ref = {id(ref): check for (_, ref), check in zip(course.iter_sources(), report.checks)}
    counts = citation_counts(report)

    def sources(refs) -> list[str]:
        lines = []
        for ref in refs:
            check = status_by_ref.get(id(ref))
            status = citation_label(check) if check else "Not checked"
            lines.append(f"> PDF page {ref.page}: “{ref.quote}” ({status})")
        return lines or ["> No source reference."]

    out = [
        f"# {meta.title}",
        "",
        f"- **Source file:** {meta.source_filename}",
        f"- **Chapter:** {meta.chapter_label} (PDF pages {meta.start_page}-{meta.end_page})",
        f"- **Generation:** {mode_label(course)}",
        f"- **Review status:** {'Approved by the teacher' if meta.review_status == 'approved' else 'Draft (not yet approved)'}",
        f"- **Created:** {meta.created_at.isoformat(timespec='seconds')}",
        f"- **Citations:** {counts['verified']} of {len(report.checks)} verified; "
        f"{counts['not_verified']} not verified; {counts['cannot_check']} could not be checked; "
        f"{counts['missing']} items without a source reference",
        "",
        f"*{TRACEABILITY_NOTE}*",
    ]
    for lesson in course.lessons:
        out += ["", f"## Lesson {lesson.number}: {lesson.title}", "", "### Learning objectives", ""]
        out += [f"- {o.text}" for o in lesson.objectives]
        out += ["", "### Explanation", "", lesson.explanation, ""] + sources(lesson.sources)
        out += ["", "### Examples"]
        for ex in lesson.examples:
            out += ["", f"**{ex.title}**", "", ex.explanation, ""] + sources(ex.sources)
        out += ["", "### Practice"]
        for i, act in enumerate(lesson.practice, 1):
            out += ["", f"{i}. {act.prompt}", "", f"   *Suggested answer:* {act.suggested_answer}", ""]
            out += sources(act.sources)
        out += ["", "### Quiz"]
        for i, q in enumerate(lesson.quiz, 1):
            out += ["", f"**Question {i}.** {q.question}", ""]
            out += [f"- {o.id}. {o.text}{'  (correct)' if o.id == q.correct_option else ''}" for o in q.options]
            out += ["", f"*Answer:* {q.correct_option}. {q.correct_answer.text}", "",
                    f"*Explanation:* {q.explanation}", ""] + sources(q.sources)
    return "\n".join(out) + "\n"
