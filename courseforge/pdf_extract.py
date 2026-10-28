"""PDF text extraction with exact page references.

Every page keeps its one-based physical PDF page number (page 1 is the first page
in the file). That number is the only page reference later stages may cite. A printed
page label (for example "xii" or "47"), when the PDF defines one, is kept separately
for display and is never used in place of the physical number.

Chapter boundaries come only from the PDF's own table of contents. When there is no
usable table of contents, the caller selects an explicit page range instead; no
boundaries are guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pymupdf

# A page needs at least this many non-whitespace characters to count as normal text.
MIN_TEXT_CHARS = 25
# Images covering at least this share of a low-text page suggest a scanned page.
SCANNED_IMAGE_COVERAGE = 0.5
DEFAULT_MAX_BYTES = 100 * 1024 * 1024


class PDFExtractionError(Exception):
    """The PDF cannot be processed. The message is safe to show to a teacher."""


class PageRangeError(ValueError):
    """A requested page range is invalid for this document."""


class PageStatus(str, Enum):
    OK = "ok"
    LOW_TEXT = "low_text"              # some text, but very little (e.g. a title page)
    EMPTY = "empty"                    # no extractable text and no large images
    LIKELY_SCANNED = "likely_scanned"  # little or no text, mostly covered by images
    ERROR = "error"                    # extraction failed for this page


@dataclass(frozen=True)
class PageText:
    page_number: int          # one-based physical PDF page number
    text: str
    label: str | None         # printed page label from the PDF, if defined
    char_count: int           # non-whitespace characters
    image_coverage: float     # share of the page area covered by images (0 to 1)
    status: PageStatus

    @property
    def has_text(self) -> bool:
        return self.status in (PageStatus.OK, PageStatus.LOW_TEXT)

    @property
    def display_name(self) -> str:
        if self.label and self.label != str(self.page_number):
            return f"PDF page {self.page_number} (printed page {self.label})"
        return f"PDF page {self.page_number}"


@dataclass(frozen=True)
class TocEntry:
    level: int
    title: str
    page_number: int          # one-based physical PDF page number


@dataclass(frozen=True)
class Chapter:
    """A section of the document whose boundaries come from the table of contents."""
    title: str
    level: int
    start_page: int
    end_page: int

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1


@dataclass
class ExtractedDocument:
    filename: str
    page_count: int
    metadata: dict[str, str]
    pages: list[PageText]
    toc: list[TocEntry]
    warnings: list[str] = field(default_factory=list)

    @property
    def has_toc(self) -> bool:
        return bool(self.toc)

    @property
    def text_page_count(self) -> int:
        return sum(p.has_text for p in self.pages)

    def page(self, page_number: int) -> PageText:
        validate_page_range(page_number, page_number, self.page_count)
        return self.pages[page_number - 1]


@dataclass
class Selection:
    """The pages a teacher chose to turn into a course."""
    label: str
    source: str               # "toc" or "page_range"
    start_page: int
    end_page: int
    pages: list[PageText]
    warnings: list[str] = field(default_factory=list)

    @property
    def text_pages(self) -> list[PageText]:
        return [p for p in self.pages if p.has_text]

    @property
    def has_usable_text(self) -> bool:
        return bool(self.text_pages)


def extract_pdf(source: bytes | str | Path, filename: str | None = None,
                *, max_bytes: int = DEFAULT_MAX_BYTES) -> ExtractedDocument:
    """Extract every page of a PDF. Raises PDFExtractionError for unusable files."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = filename or path.name
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise PDFExtractionError(f"Could not read the file: {exc.strerror or exc}.") from exc
    else:
        data = bytes(source)
    filename = filename or "uploaded.pdf"

    if not data:
        raise PDFExtractionError("The file is empty.")
    if len(data) > max_bytes:
        raise PDFExtractionError(
            f"The file is too large ({len(data) / 1_048_576:.1f} MB); "
            f"the limit is {max_bytes / 1_048_576:.0f} MB.")
    if b"%PDF-" not in data[:1024]:
        raise PDFExtractionError("This file is not a PDF.")

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types for damaged files
        raise PDFExtractionError("The PDF could not be opened; it may be damaged.") from exc

    with doc:
        if doc.needs_pass and not doc.authenticate(""):
            raise PDFExtractionError(
                "The PDF is password protected. Please upload an unlocked copy.")
        if doc.page_count == 0:
            raise PDFExtractionError("The PDF has no pages.")

        pages = [_extract_page(doc, i) for i in range(doc.page_count)]
        try:
            raw_toc = doc.get_toc(simple=True)
        except Exception:
            raw_toc = []
            toc_warnings = ["The table of contents could not be read; select a page range instead."]
        else:
            toc_warnings = []
        toc, parse_warnings = parse_toc(raw_toc, doc.page_count)
        metadata = {k: v.strip() for k, v in (doc.metadata or {}).items()
                    if isinstance(v, str) and v.strip()}
        page_count = doc.page_count

    warnings = _page_warnings(pages) + toc_warnings + parse_warnings
    return ExtractedDocument(filename=filename, page_count=page_count, metadata=metadata,
                             pages=pages, toc=toc, warnings=warnings)


def _extract_page(doc: pymupdf.Document, index: int) -> PageText:
    page_number = index + 1
    try:
        page = doc[index]
        text = _clean_text(page.get_text("text"))
        label = page.get_label() or None
        coverage = _image_coverage(page)
    except Exception:
        return PageText(page_number, "", None, 0, 0.0, PageStatus.ERROR)

    chars = len(re.sub(r"\s", "", text))
    if chars >= MIN_TEXT_CHARS:
        status = PageStatus.OK
    elif coverage >= SCANNED_IMAGE_COVERAGE:
        status = PageStatus.LIKELY_SCANNED
    elif chars > 0:
        status = PageStatus.LOW_TEXT
    else:
        status = PageStatus.EMPTY
    return PageText(page_number, text, label, chars, coverage, status)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _image_coverage(page: pymupdf.Page) -> float:
    page_area = abs(page.rect)
    if not page_area:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        bbox = pymupdf.Rect(info["bbox"]) & page.rect  # clip to the visible page
        covered += abs(bbox)
    return min(covered / page_area, 1.0)


def _page_warnings(pages: list[PageText]) -> list[str]:
    def numbers(status: PageStatus) -> list[int]:
        return [p.page_number for p in pages if p.status == status]

    warnings = []
    scanned = numbers(PageStatus.LIKELY_SCANNED)
    if scanned:
        warnings.append(
            f"{_plural(scanned)} {format_page_list(scanned)} "
            f"{'appears' if len(scanned) == 1 else 'appear'} to be scanned images without "
            "extractable text. Text recognition (OCR) is not supported, so "
            f"{'it' if len(scanned) == 1 else 'they'} will be skipped.")
        if len(scanned) > len(pages) / 2:
            warnings.append("Most of this PDF appears to be scanned. Results will be incomplete; "
                            "a PDF with selectable text is recommended.")
    empty = numbers(PageStatus.EMPTY)
    if empty:
        warnings.append(f"{_plural(empty)} {format_page_list(empty)} "
                        f"{'has' if len(empty) == 1 else 'have'} no extractable text.")
    failed = numbers(PageStatus.ERROR)
    if failed:
        warnings.append(f"{_plural(failed)} {format_page_list(failed)} could not be read.")
    return warnings


def _plural(page_numbers: list[int]) -> str:
    return "Page" if len(page_numbers) == 1 else "Pages"


def format_page_list(page_numbers: list[int]) -> str:
    """Format [1, 2, 3, 7] as "1-3, 7"."""
    if not page_numbers:
        return ""
    nums = sorted(set(page_numbers))
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(parts)


def parse_toc(raw_toc: list, page_count: int) -> tuple[list[TocEntry], list[str]]:
    """Keep only table-of-contents entries that point to a real page."""
    entries, skipped = [], 0
    for item in raw_toc:
        try:
            level, title, page = int(item[0]), str(item[1]).strip(), int(item[2])
        except (TypeError, ValueError, IndexError):
            skipped += 1
            continue
        if level < 1 or not title or not 1 <= page <= page_count:
            skipped += 1
            continue
        entries.append(TocEntry(level, title, page))
    warnings = []
    if skipped:
        warnings.append(f"{skipped} table of contents "
                        f"{'entry was' if skipped == 1 else 'entries were'} skipped because "
                        "they did not point to a valid page.")
    return entries, warnings


def list_chapters(doc: ExtractedDocument, max_level: int | None = None) -> list[Chapter]:
    """Chapters and sections from the table of contents, with page ranges.

    A section ends on the page before the next entry at the same or a higher level
    starts, or on the last page of the document. Entries that start on the same page
    share that page. Returns an empty list when there is no usable table of contents.
    """
    chapters = []
    for i, entry in enumerate(doc.toc):
        if max_level is not None and entry.level > max_level:
            continue
        end = doc.page_count
        for nxt in doc.toc[i + 1:]:
            if nxt.level <= entry.level and nxt.page_number >= entry.page_number:
                end = max(entry.page_number, nxt.page_number - 1)
                break
        chapters.append(Chapter(entry.title, entry.level, entry.page_number, end))
    return chapters


def validate_page_range(start: int, end: int, page_count: int) -> None:
    for name, value in (("Start page", start), ("End page", end)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise PageRangeError(f"{name} must be a whole number.")
    if start < 1:
        raise PageRangeError("Start page must be 1 or higher.")
    if end > page_count:
        raise PageRangeError(f"End page must be {page_count} or lower; the PDF has {page_count} pages.")
    if start > end:
        raise PageRangeError("Start page must not be after the end page.")


def select_page_range(doc: ExtractedDocument, start: int, end: int,
                      label: str | None = None, source: str = "page_range") -> Selection:
    validate_page_range(start, end, doc.page_count)
    pages = doc.pages[start - 1:end]
    if label is None:
        label = f"Page {start}" if start == end else f"Pages {start}-{end}"
    selection = Selection(label=label, source=source, start_page=start, end_page=end, pages=pages)
    skipped = [p.page_number for p in pages if not p.has_text]
    if not selection.has_usable_text:
        selection.warnings.append("The selected pages contain no extractable text.")
    elif skipped:
        selection.warnings.append(
            f"{len(skipped)} of {len(pages)} selected pages have no usable text "
            f"(pages {format_page_list(skipped)}) and will be skipped.")
    return selection


def select_chapter(doc: ExtractedDocument, chapter: Chapter) -> Selection:
    """Select a chapter that came from this document's table of contents."""
    if chapter not in list_chapters(doc):
        raise PageRangeError("That chapter is not in this document's table of contents.")
    return select_page_range(doc, chapter.start_page, chapter.end_page,
                             label=chapter.title, source="toc")
