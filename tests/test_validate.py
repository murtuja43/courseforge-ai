"""Tests for citation validation in courseforge.validate."""

import pymupdf

from courseforge.pdf_extract import PageStatus, PageText, Selection, extract_pdf, select_page_range
from courseforge.schema import Course
from courseforge.validate import TRACEABILITY_NOTE, CitationStatus, check_citations, normalize_text
from test_schema import course_dict

PAGE_TEXT = {
    1: "Photosynthesis converts light energy into chemical energy stored in glucose.",
    2: "Chlorophyll is the green pigment that absorbs light inside the chloroplast.",
    3: "The Calvin cycle uses carbon dioxide to build sugars in the stroma.",
}


def page(n, text=None, status=PageStatus.OK):
    text = PAGE_TEXT.get(n, "") if text is None else text
    return PageText(n, text, None, len(text.replace(" ", "")), 0.0, status)


def selection(start=1, end=3, pages=None):
    pages = pages or [page(n) for n in range(start, end + 1)]
    return Selection(label="Chapter 3", source="toc", start_page=start, end_page=end, pages=pages)


def course_with(lesson_sources=None, quiz_sources=None, start=1, end=3):
    data = course_dict()
    data["metadata"].update(start_page=start, end_page=end)
    if lesson_sources is not None:
        data["lessons"][0]["sources"] = lesson_sources
    if quiz_sources is not None:
        data["lessons"][0]["quiz"][0]["sources"] = quiz_sources
    return Course.model_validate(data)


def checks_at(report, location):
    return [c for c in report.checks if c.location == location]


# ---------------------------------------------------------------- valid citations

def test_valid_citations_are_verified():
    report = check_citations(course_with(), selection())
    assert report.all_verified
    assert report.verified_count == len(report.checks) > 0
    assert all(c.status == CitationStatus.VERIFIED for c in report.checks)


def test_multiple_citations_for_one_lesson_are_each_checked():
    sources = [{"page": 1, "quote": "converts light energy into chemical energy"},
               {"page": 2, "quote": "the green pigment that absorbs light"},
               {"page": 3, "quote": "build sugars in the stroma"}]
    report = check_citations(course_with(lesson_sources=sources), selection())
    lesson_checks = checks_at(report, "Lesson 1")
    assert [(c.page, c.verified) for c in lesson_checks] == [(1, True), (2, True), (3, True)]


def test_one_bad_citation_among_good_ones_is_caught():
    sources = [{"page": 1, "quote": "converts light energy into chemical energy"},
               {"page": 2, "quote": "converts light energy into chemical energy"}]  # wrong page
    report = check_citations(course_with(lesson_sources=sources), selection())
    assert [c.verified for c in checks_at(report, "Lesson 1")] == [True, False]
    assert not report.all_verified


# ---------------------------------------------------------------- invalid citations

def test_quote_not_on_the_cited_page():
    sources = [{"page": 1, "quote": "mitochondria are the powerhouse of the cell"}]
    report = check_citations(course_with(lesson_sources=sources), selection())
    [check] = checks_at(report, "Lesson 1")
    assert check.status == CitationStatus.QUOTE_NOT_FOUND
    assert not check.verified
    assert check.message.startswith("quote not found on page 1:")


def test_paraphrased_quote_is_not_accepted():
    sources = [{"page": 1, "quote": "turns light energy into chemical energy"}]
    report = check_citations(course_with(lesson_sources=sources), selection())
    assert checks_at(report, "Lesson 1")[0].status == CitationStatus.QUOTE_NOT_FOUND


def test_citation_outside_the_selected_chapter():
    sources = [{"page": 5, "quote": "converts light energy into chemical energy"}]
    report = check_citations(course_with(lesson_sources=sources), selection(), document_page_count=10)
    [check] = checks_at(report, "Lesson 1")
    assert check.status == CitationStatus.OUTSIDE_SELECTION
    assert "outside the selected pages 1-3" in check.message


def test_citation_to_a_nonexistent_page():
    sources = [{"page": 99, "quote": "converts light energy into chemical energy"}]
    report = check_citations(course_with(lesson_sources=sources), selection(), document_page_count=10)
    [check] = checks_at(report, "Lesson 1")
    assert check.status == CitationStatus.PAGE_NOT_IN_DOCUMENT
    assert "page 99 does not exist; the PDF has 10 pages" in check.message


def test_nonexistent_page_without_page_count_is_still_rejected():
    sources = [{"page": 99, "quote": "converts light energy into chemical energy"}]
    report = check_citations(course_with(lesson_sources=sources), selection())
    assert checks_at(report, "Lesson 1")[0].status == CitationStatus.OUTSIDE_SELECTION


def test_citation_to_a_page_without_text_is_not_verified():
    pages = [page(1), page(2, "", PageStatus.LIKELY_SCANNED), page(3)]
    sources = [{"page": 2, "quote": "the green pigment that absorbs light"}]
    report = check_citations(course_with(lesson_sources=sources), selection(pages=pages))
    assert checks_at(report, "Lesson 1")[0].status == CitationStatus.PAGE_HAS_NO_TEXT


def test_course_page_range_must_match_the_selection():
    report = check_citations(course_with(start=1, end=4), selection())
    assert report.course_errors == ["course says pages 1-4, but the selected pages are 1-3."]
    assert not report.all_verified


def test_problems_in_quiz_items_are_located():
    report = check_citations(course_with(quiz_sources=[{"page": 3, "quote": "glucose is a protein"}]),
                             selection())
    [check] = checks_at(report, "Lesson 1, quiz question 1")
    assert not check.verified


# ---------------------------------------------------------------- reporting

def test_report_summary_is_clear():
    sources = [{"page": 7, "quote": "converts light energy into chemical energy"}]
    report = check_citations(course_with(lesson_sources=sources), selection())
    summary = report.summary()
    assert summary.startswith(f"{report.verified_count} of {len(report.checks)} citations verified.")
    assert "Lesson 1: page 7 is outside the selected pages 1-3." in summary
    assert "No source reference: Lesson 1, practice activity 1" in summary
    assert summary.endswith(TRACEABILITY_NOTE)


def test_verified_report_still_states_that_content_is_not_checked():
    assert "do not confirm that explanations" in check_citations(course_with(), selection()).summary()


def test_empty_report_is_not_all_verified():
    from courseforge.validate import CitationReport
    assert not CitationReport(checks=[]).all_verified


# ---------------------------------------------------------------- quote matching

def test_normalization_handles_pdf_text_artifacts():
    page_text = "The eﬃcient “leaf” pro-\ncess   runs\nduring  DAYLIGHT – mostly."
    assert normalize_text('efficient "leaf" process runs during daylight - mostly') in normalize_text(page_text)


def test_quote_matching_works_on_real_extracted_pdf_text():
    doc = pymupdf.open()
    for n in (1, 2, 3):
        p = doc.new_page()
        p.insert_textbox(pymupdf.Rect(72, 72, 300, 400), PAGE_TEXT[n], fontsize=11)  # wraps lines
    extracted = extract_pdf(doc.tobytes())
    sel = select_page_range(extracted, 1, 3)
    sources = [{"page": 2, "quote": "Chlorophyll is the green pigment that absorbs light inside the chloroplast."}]
    report = check_citations(course_with(lesson_sources=sources), sel, extracted.page_count)
    assert "\n" in sel.pages[1].text  # the passage really was wrapped across lines
    assert checks_at(report, "Lesson 1")[0].verified
