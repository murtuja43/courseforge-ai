"""Structure and citation validation for generated courses.

Citation checks confirm traceability only: that each cited page is in the selected
chapter and that the quoted passage really appears in the text extracted from that
page. A matching quotation does not prove that the explanation around it is correct;
teacher review is still required.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

from pydantic import ValidationError

from courseforge.pdf_extract import Selection
from courseforge.schema import Course

TRACEABILITY_NOTE = ("Citation checks confirm that quoted passages appear on the cited pages. "
                     "They do not confirm that explanations, examples or answers are correct; "
                     "teacher review is required.")


class CitationStatus(str, Enum):
    VERIFIED = "verified"
    PAGE_NOT_IN_DOCUMENT = "page_not_in_document"
    OUTSIDE_SELECTION = "outside_selection"
    PAGE_HAS_NO_TEXT = "page_has_no_text"
    QUOTE_NOT_FOUND = "quote_not_found"


@dataclass(frozen=True)
class CitationCheck:
    location: str
    page: int
    quote: str
    status: CitationStatus
    message: str

    @property
    def verified(self) -> bool:
        return self.status == CitationStatus.VERIFIED


@dataclass
class CitationReport:
    checks: list[CitationCheck]
    uncited: list[str] = field(default_factory=list)     # items with no source reference
    course_errors: list[str] = field(default_factory=list)

    @property
    def problems(self) -> list[CitationCheck]:
        return [c for c in self.checks if not c.verified]

    @property
    def verified_count(self) -> int:
        return sum(c.verified for c in self.checks)

    @property
    def all_verified(self) -> bool:
        return bool(self.checks) and not self.problems and not self.course_errors

    def summary(self) -> str:
        lines = [f"{self.verified_count} of {len(self.checks)} citations verified."]
        lines += [f"Course: {e}" for e in self.course_errors]
        lines += [f"{c.location}: {c.message}" for c in self.problems]
        if self.uncited:
            lines.append("No source reference: " + "; ".join(self.uncited) + ".")
        lines.append(TRACEABILITY_NOTE)
        return "\n".join(lines)


def parse_course(data: dict | str) -> tuple[Course | None, list[str]]:
    """Validate raw course data (a dict or JSON text). Returns the course or readable errors."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            return None, [f"Not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."]
    try:
        return Course.model_validate(data), []
    except ValidationError as exc:
        return None, [_format_error(err) for err in exc.errors()]


def _format_error(err: dict) -> str:
    parts = []
    for item in err["loc"]:
        if isinstance(item, int):
            parts[-1:] = [f"{parts[-1]} {item + 1}" if parts else f"item {item + 1}"]
        else:
            parts.append(str(item))
    where = " > ".join(parts) or "course"
    message = err["msg"].removeprefix("Value error, ")
    return f"{where}: {message}"


def normalize_text(text: str) -> str:
    """Normalize text for quote matching: ligatures, quote marks, hyphenation, spacing, case."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")                                   # soft hyphen
    text = re.sub(r"[‘’‚‛′]", "'", text)
    text = re.sub(r"[“”„‟″]", '"', text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)                  # word broken across lines
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def check_citations(course: Course, selection: Selection,
                    document_page_count: int | None = None) -> CitationReport:
    """Check every source reference in the course against the selected pages."""
    pages = {p.page_number: p for p in selection.pages}
    normalized: dict[int, str] = {}
    checks = []
    for location, ref in course.iter_sources():
        page = pages.get(ref.page)
        if document_page_count is not None and ref.page > document_page_count:
            status = CitationStatus.PAGE_NOT_IN_DOCUMENT
            message = f"page {ref.page} does not exist; the PDF has {document_page_count} pages."
        elif page is None:
            status = CitationStatus.OUTSIDE_SELECTION
            message = (f"page {ref.page} is outside the selected pages "
                       f"{selection.start_page}-{selection.end_page}.")
        elif not page.has_text:
            status = CitationStatus.PAGE_HAS_NO_TEXT
            message = f"page {ref.page} has no extractable text, so the quote cannot be checked."
        else:
            if ref.page not in normalized:
                normalized[ref.page] = normalize_text(page.text)
            if normalize_text(ref.quote) in normalized[ref.page]:
                status, message = CitationStatus.VERIFIED, f"quote found on page {ref.page}."
            else:
                status = CitationStatus.QUOTE_NOT_FOUND
                message = f"quote not found on page {ref.page}: \"{_shorten(ref.quote)}\""
        checks.append(CitationCheck(location, ref.page, ref.quote, status, message))

    course_errors = []
    meta = course.metadata
    if (meta.start_page, meta.end_page) != (selection.start_page, selection.end_page):
        course_errors.append(
            f"course says pages {meta.start_page}-{meta.end_page}, but the selected pages are "
            f"{selection.start_page}-{selection.end_page}.")
    return CitationReport(checks=checks, uncited=course.uncited_items(), course_errors=course_errors)


def _shorten(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
