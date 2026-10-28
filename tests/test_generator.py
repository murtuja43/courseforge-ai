"""Tests for the offline demo generator.

The fixture chapter is entirely fictional ("glowmoss" on the imaginary planet Orvane);
it is not taken from any real textbook.
"""

import re
import socket
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import pymupdf
import pytest

import courseforge.generator as generator
from courseforge.generator import (
    DEMO_NOTICE, InsufficientSourceError, InvalidSelectionError, extract_sentences,
    generate_demo_course,
)
from courseforge.pdf_extract import PageStatus, PageText, Selection, extract_pdf, select_page_range
from courseforge.validate import check_citations, normalize_text

FICTIONAL_CHAPTER = [
    "Chapter 2: The Glowmoss of Orvane\n"
    "Glowmoss is a fictional plant that grows on the cliffs of the imaginary planet Orvane. "
    "The moss produces a soft blue light at night by storing sunlight during the day. "
    "Each moss colony forms a thick carpet of tiny leaves called lumenfronds. "
    "The lumenfronds absorb sunlight and convert it into stored glow energy. "
    "For example, a colony that receives six hours of sunlight can glow for the whole night.",

    "Glowmoss spreads by releasing silver spores into the cliff winds. "
    "The spores drift for many days before they settle on wet stone. "
    "A spore only grows when the stone stays damp for a full week. "
    "Young colonies glow weakly because their lumenfronds are still small. "
    "Farmers on Orvane collect the spores to start new colonies near their homes.",

    "The glow of the moss attracts night insects called dimwings. "
    "The dimwings carry spores from one cliff to another while they feed. "
    "In return, the moss provides the dimwings with a sweet nectar. "
    "Scientists in the story describe this partnership as a form of mutual support. "
    "Without the dimwings, the moss would spread much more slowly across the cliffs.",

    "Cold winters reduce the amount of stored glow energy in each colony. "
    "During long winters, the moss enters a resting state and stops releasing spores. "
    "The colony wakes again when the cliffs warm and the stone becomes damp. "
    "A healthy colony can survive many winters by resting through the coldest months. "
    "Farmers protect their colonies by covering them with woven grass during storms.",
]
FIXED_TIME = datetime(2026, 11, 1, tzinfo=timezone.utc)


def make_pdf(page_texts, scanned_pages=()):
    """Pages of wrapped text, like a real textbook. None makes a blank page."""
    doc = pymupdf.open()
    for i, text in enumerate(page_texts, start=1):
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(72, 72, 400, 760), text, fontsize=11)
        if i in scanned_pages:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
            pix.set_rect(pix.irect, (210, 210, 210))
            page.insert_image(page.rect, pixmap=pix)
    return doc.tobytes()


def chapter_selection(texts=FICTIONAL_CHAPTER, scanned_pages=(), prefix_pages=0):
    """Extract a PDF and select the chapter pages, which may start after other pages."""
    filler = ["Front matter page."] * prefix_pages
    doc = extract_pdf(make_pdf(filler + list(texts), scanned_pages), "orvane-fiction.pdf")
    start = prefix_pages + 1
    return doc, select_page_range(doc, start, prefix_pages + len(texts), label="Chapter 2: Glowmoss")


def generate(selection, doc):
    return generate_demo_course(selection, doc.filename, document_page_count=doc.page_count,
                                created_at=FIXED_TIME)


@pytest.fixture(scope="module")
def result_and_selection():
    doc, sel = chapter_selection(prefix_pages=3)  # chapter on PDF pages 4-7
    return generate(sel, doc), sel


# ---------------------------------------------------------------- a valid chapter

def test_valid_chapter_produces_a_verified_five_lesson_course(result_and_selection):
    result, _ = result_and_selection
    course = result.course
    assert [lesson.number for lesson in course.lessons] == [1, 2, 3, 4, 5]
    assert len({lesson.title.casefold() for lesson in course.lessons}) == 5
    assert result.verified
    assert result.citations.verified_count == len(result.citations.checks) > 0
    for lesson in course.lessons:
        assert lesson.objectives and lesson.explanation and lesson.examples and lesson.practice
        assert lesson.quiz and lesson.sources


def test_metadata_marks_demo_mode_draft_and_no_ai(result_and_selection):
    result, _ = result_and_selection
    meta = result.course.metadata
    assert meta.generation_mode == "demo"
    assert meta.model is None
    assert meta.review_status == "draft"
    assert (meta.start_page, meta.end_page) == (4, 7)
    assert meta.source_filename == "orvane-fiction.pdf"
    assert meta.chapter_label == "Chapter 2: Glowmoss"
    assert "No AI model was used" in result.status_message
    assert result.status_message.startswith(DEMO_NOTICE)


def test_output_round_trips_through_the_schema(result_and_selection):
    from courseforge.validate import parse_course
    result, _ = result_and_selection
    again, errors = parse_course(result.course.model_dump_json())
    assert errors == [] and again == result.course


def test_every_quote_appears_on_its_cited_physical_page(result_and_selection):
    result, sel = result_and_selection
    page_text = {p.page_number: normalize_text(p.text) for p in sel.pages}
    refs = list(result.course.iter_sources())
    assert {ref.page for _, ref in refs} == {4, 5, 6, 7}  # physical pages, not chapter-relative
    for location, ref in refs:
        assert normalize_text(ref.quote) in page_text[ref.page], location


def test_explanations_only_quote_source_text(result_and_selection):
    result, sel = result_and_selection
    all_text = normalize_text(" ".join(p.text for p in sel.pages))
    for lesson in result.course.lessons:
        quoted = re.findall(r"“(.+?)” \(page (\d+)\)", lesson.explanation)
        assert quoted
        for text, page in quoted:
            assert normalize_text(text) in normalize_text(sel.pages[int(page) - 4].text)
            assert normalize_text(text) in all_text


def test_heading_is_not_glued_to_the_first_sentence():
    _, sel = chapter_selection()
    first = extract_sentences(sel)[0]
    assert first.text.startswith("Glowmoss is a fictional plant")


def test_quiz_questions_are_source_based_completions(result_and_selection):
    result, _ = result_and_selection
    for lesson in result.course.lessons:
        for q in lesson.quiz:
            answer = q.correct_answer.text
            source = q.sources[0].quote
            assert q.question.startswith("Source-based completion:")
            assert "_____" in q.question
            assert re.search(rf"\b{answer}\b", source, re.IGNORECASE)
            assert not re.search(rf"\b{answer}\b", q.question, re.IGNORECASE)  # answer is blanked
            for option in q.options:
                if option.id != q.correct_option:  # distractors never appear in the sentence
                    assert not re.search(rf"\b{option.text}\b", source, re.IGNORECASE)
            stems = [o.text[:4] for o in q.options]  # no near-duplicates such as colony/colonies
            assert len(set(stems)) == len(stems), [o.text for o in q.options]
            assert source in q.explanation


def test_real_example_sentence_is_used_when_present(result_and_selection):
    result, _ = result_and_selection
    examples = [ex for lesson in result.course.lessons for ex in lesson.examples
                if ex.title.startswith("Example from the text")]
    assert len(examples) == 1  # the fixture has exactly one "For example" sentence
    assert "For example, a colony that receives six hours" in examples[0].explanation
    assert examples[0].sources[0].page == 4


def test_missing_examples_are_labeled_and_reported(result_and_selection):
    result, _ = result_and_selection
    discussion = [ex for lesson in result.course.lessons for ex in lesson.examples
                  if ex.title.startswith("Source passage for discussion")]
    assert discussion
    assert all("do not mark a worked example" in ex.explanation for ex in discussion)
    assert any("no worked example was found" in n for n in result.notices)


# ---------------------------------------------------------------- determinism

def test_same_input_gives_identical_output():
    doc, sel = chapter_selection()
    first = generate(sel, doc).course.model_dump()
    second = generate(sel, doc).course.model_dump()
    assert first == second


def test_output_is_identical_apart_from_timestamp_when_none_given():
    doc, sel = chapter_selection()
    a = generate_demo_course(sel, doc.filename).course.model_dump(exclude={"metadata": {"created_at"}})
    b = generate_demo_course(sel, doc.filename).course.model_dump(exclude={"metadata": {"created_at"}})
    assert a == b


# ---------------------------------------------------------------- validator integration

def test_tampered_quote_is_not_verified(result_and_selection):
    result, sel = result_and_selection
    course = result.course.model_copy(deep=True)
    course.lessons[2].sources[0].quote = "The moss is made of solid gold and sings."
    report = check_citations(course, sel)
    assert not report.all_verified
    assert report.problems[0].location == "Lesson 3"


# ---------------------------------------------------------------- insufficient or unusable text

def test_empty_chapter_is_rejected():
    doc, sel = chapter_selection([None, None])
    with pytest.raises(InsufficientSourceError, match="no extractable text"):
        generate(sel, doc)


def test_nearly_empty_chapter_is_rejected_with_a_count():
    doc, sel = chapter_selection(["Glowmoss spreads by releasing silver spores into the cliff winds. "
                                  "The spores drift for many days before they settle on wet stone."])
    with pytest.raises(InsufficientSourceError, match=r"contain 2 usable sentence\(s\); at least 10"):
        generate(sel, doc)


def test_scanned_chapter_is_rejected_and_says_why():
    doc, sel = chapter_selection([None, None, None], scanned_pages={1, 2, 3})
    assert all(p.status == PageStatus.LIKELY_SCANNED for p in sel.pages)
    with pytest.raises(InsufficientSourceError, match="scanned images"):
        generate(sel, doc)


def test_scanned_page_inside_a_usable_chapter_is_skipped_and_reported():
    texts = FICTIONAL_CHAPTER[:2] + [None] + FICTIONAL_CHAPTER[2:]
    doc, sel = chapter_selection(texts, scanned_pages={3})
    result = generate(sel, doc)
    assert result.verified
    assert 3 not in {ref.page for _, ref in result.course.iter_sources()}
    assert any("will be skipped" in n for n in result.notices)


def test_text_without_repeated_key_terms_cannot_make_a_quiz():
    # Ten sentences of unique invented words: nothing repeats, so no reliable quiz item exists.
    words = ("".join(p) for p in product("bdfgk", "aeiou", "lmnrt", "aeiou", "sz"))
    sentences = [" ".join(next(words) for _ in range(8)).capitalize() + "." for _ in range(10)]
    doc, sel = chapter_selection([" ".join(sentences[:5]), " ".join(sentences[5:])])
    with pytest.raises(InsufficientSourceError, match="no reliable quiz question"):
        generate(sel, doc)


# ---------------------------------------------------------------- invalid selections

def text_page(n):
    return PageText(n, FICTIONAL_CHAPTER[0], None, 300, 0.0, PageStatus.OK)


def test_missing_source_pages_are_rejected():
    sel = Selection("Chapter", "page_range", 1, 3, [text_page(1), text_page(3)])
    with pytest.raises(InvalidSelectionError, match="missing for pages 2"):
        generate_demo_course(sel, "book.pdf")


@pytest.mark.parametrize("start,end,pages,count,message", [
    (3, 1, [], None, "Invalid page range 3-1"),
    (0, 1, [0, 1], None, "Invalid page range 0-1"),
    (1, 2, [1, 2, 3], None, "outside the selected range"),
    (1, 2, [2, 1], None, "duplicated or out of order"),
    (1, 2, [1, 2], 1, "past the end of the PDF"),
])
def test_invalid_page_ranges_are_rejected(start, end, pages, count, message):
    sel = Selection("Chapter", "page_range", start, end, [text_page(n) for n in pages])
    with pytest.raises(InvalidSelectionError, match=message):
        generate_demo_course(sel, "book.pdf", document_page_count=count)


# ---------------------------------------------------------------- offline

def test_generator_makes_no_network_calls(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    doc, sel = chapter_selection()
    assert generate(sel, doc).verified


def test_generator_does_not_import_network_or_llm_libraries():
    source = Path(generator.__file__).read_text(encoding="utf-8")
    for name in ("anthropic", "openai", "requests", "httpx", "urllib", "socket"):
        assert not re.search(rf"^\s*(import|from)\s+{name}\b", source, re.MULTILINE), name
