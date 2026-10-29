"""Tests for courseforge.ui_helpers and the fictional sample textbook."""

import json
import socket

import pytest

from courseforge import ui_helpers as ui
from courseforge.generator import generate_demo_course
from courseforge.llm import LLMConfig
from courseforge.pdf_extract import Chapter, PageStatus, PageText, Selection, extract_pdf, list_chapters, select_chapter
from courseforge.sample import SAMPLE_FILENAME, sample_pdf_bytes
from courseforge.validate import TRACEABILITY_NOTE, CitationStatus
from test_generator import chapter_selection
from test_llm import CONFIG, KEY_DETAILS, QUOTA_DETAILS, SECRET, FakeClient, api_error, course_json


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("a real network connection was attempted")
    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(scope="module")
def sample():
    doc = extract_pdf(sample_pdf_bytes(), SAMPLE_FILENAME)
    sel = select_chapter(doc, list_chapters(doc)[0])
    result = generate_demo_course(sel, doc.filename, document_page_count=doc.page_count)
    return doc, sel, result.course.model_dump(mode="json")


# ---------------------------------------------------------------- sample textbook

def test_sample_textbook_has_a_real_table_of_contents():
    doc = extract_pdf(sample_pdf_bytes(), SAMPLE_FILENAME)
    assert doc.page_count == 8 and doc.text_page_count == 8 and doc.warnings == []
    assert [(c.title, c.start_page, c.end_page) for c in list_chapters(doc)] == [
        ("Chapter 1: The Glowmoss of Orvane", 2, 5), ("Chapter 2: Dimwings and the Night Cliffs", 6, 8)]
    assert "fictional" in doc.metadata["title"]


@pytest.mark.parametrize("index", [0, 1])
def test_both_sample_chapters_build_verified_demo_courses(index):
    doc = extract_pdf(sample_pdf_bytes(), SAMPLE_FILENAME)
    sel = select_chapter(doc, list_chapters(doc)[index])
    assert generate_demo_course(sel, doc.filename, document_page_count=doc.page_count).verified


def test_chapter_label():
    assert ui.chapter_label(Chapter("Chapter 1", 1, 2, 5)) == "Chapter 1 (pages 2-5)"
    assert ui.chapter_label(Chapter("1.1 Intro", 2, 3, 3)) == " 1.1 Intro (page 3)"


# ---------------------------------------------------------------- editing

def test_editable_fields_exclude_numbers_and_option_ids(sample):
    _, _, data = sample
    paths = [path for path, _ in ui.editable_fields(data)]
    assert ("lessons", 0, "title") in paths
    assert ("lessons", 0, "quiz", 0, "correct_option") in paths
    assert ("lessons", 0, "sources", 0, "page") in paths
    assert not any(p[-1] in ("number", "id") for p in paths)


def test_seeding_never_overwrites_existing_edits(sample):
    _, _, data = sample
    state = {}
    ui.seed_edit_state(state, 1, data)
    key = ui.widget_key(1, ("lessons", 0, "title"))
    state[key] = "Teacher's title"
    ui.seed_edit_state(state, 1, data)
    assert state[key] == "Teacher's title"


def test_edits_in_several_lessons_are_all_collected(sample):
    _, _, data = sample
    state = {}
    ui.seed_edit_state(state, 1, data)
    state[ui.widget_key(1, ("lessons", 0, "explanation"))] = "Edited in lesson 1."
    state[ui.widget_key(1, ("lessons", 4, "quiz", 0, "options", 1, "text"))] = "edited option"
    edited = ui.collect_edits(state, 1, data)
    assert edited["lessons"][0]["explanation"] == "Edited in lesson 1."
    assert edited["lessons"][4]["quiz"][0]["options"][1]["text"] == "edited option"
    assert data["lessons"][0]["explanation"] != "Edited in lesson 1."  # the original is not mutated


def test_source_paths_match_the_validator_order(sample):
    doc, sel, data = sample
    evaluation = ui.evaluate_course(data, sel, doc.page_count)
    paths = list(ui.source_paths(data))
    assert len(paths) == len(evaluation.report.checks)
    for path, check in zip(paths, evaluation.report.checks):
        node = data
        for part in path:
            node = node[part]
        assert (node["page"], node["quote"]) == (check.page, check.quote)


def test_edited_quote_is_rechecked_and_not_verified(sample):
    doc, sel, data = sample
    state = {}
    ui.seed_edit_state(state, 1, data)
    path = ("lessons", 1, "sources", 0)
    state[ui.widget_key(1, path + ("quote",))] = "A sentence that is not in the textbook at all."
    evaluation = ui.evaluate_course(ui.collect_edits(state, 1, data), sel, doc.page_count)
    assert evaluation.checks_by_path[path].status == CitationStatus.QUOTE_NOT_FOUND
    assert not evaluation.report.all_verified


def test_invalid_edit_is_reported_and_blocks_evaluation(sample):
    doc, sel, data = sample
    state = {}
    ui.seed_edit_state(state, 1, data)
    state[ui.widget_key(1, ("lessons", 2, "title"))] = "   "
    evaluation = ui.evaluate_course(ui.collect_edits(state, 1, data), sel, doc.page_count)
    assert evaluation.course is None and evaluation.report is None
    assert evaluation.errors[0].startswith("lessons 3 > title:")


# ---------------------------------------------------------------- approval

def test_approval_holds_only_while_content_is_unchanged(sample):
    doc, sel, data = sample
    course = ui.evaluate_course(data, sel, doc.page_count).course
    assert ui.with_review_status(course, None).metadata.review_status == "draft"
    approved_fp = ui.fingerprint(course)
    assert ui.with_review_status(course, approved_fp).metadata.review_status == "approved"
    state = {}
    ui.seed_edit_state(state, 1, data)
    state[ui.widget_key(1, ("lessons", 0, "title"))] = "Changed after approval"
    changed = ui.evaluate_course(ui.collect_edits(state, 1, data), sel, doc.page_count).course
    assert ui.with_review_status(changed, approved_fp).metadata.review_status == "draft"


# ---------------------------------------------------------------- citation reporting

def text_page(n, text="Glowmoss produces light at night by storing sunlight.", status=PageStatus.OK):
    return PageText(n, text, None, 40, 0.0, status)


def test_citation_rows_label_every_status_honestly(sample):
    doc, sel, data = sample
    pages = [text_page(2), text_page(3, "", PageStatus.LIKELY_SCANNED), text_page(4), text_page(5)]
    selection = Selection("Chapter", "toc", 2, 5, pages)
    edited = json.loads(json.dumps(data))
    lesson = edited["lessons"][0]
    lesson["sources"] = [{"page": 2, "quote": "Glowmoss produces light at night"},   # verified
                         {"page": 2, "quote": "This sentence is invented here."},    # not found
                         {"page": 3, "quote": "Anything on a scanned page."},        # no text
                         {"page": 7, "quote": "Glowmoss produces light at night"},   # outside
                         {"page": 99, "quote": "Glowmoss produces light at night"}]  # no such page
    lesson["examples"][0]["sources"] = []                                            # missing
    report = ui.evaluate_course(edited, selection, doc.page_count).report
    rows = [r for r in ui.citation_rows(report) if r["Location"] in ("Lesson 1", "Lesson 1, example 1")]
    assert [r["Status"] for r in rows] == [
        "Verified", "Not verified: quote not found on the page", "Cannot check: page has no usable text",
        "Not verified: outside the selected chapter", "Not verified: page does not exist",
        "Missing: no source reference"]
    counts = ui.citation_counts(report)  # other lessons also cite page 3, which is scanned here
    assert counts["missing"] == 1 and counts["cannot_check"] >= 1 and counts["not_verified"] >= 3
    assert sum(counts[k] for k in ("verified", "not_verified", "cannot_check")) == len(report.checks)


# ---------------------------------------------------------------- export

def edited_course(sample, approve=False):
    doc, sel, data = sample
    state = {}
    ui.seed_edit_state(state, 1, data)
    state[ui.widget_key(1, ("lessons", 0, "title"))] = "Teacher-edited title"
    state[ui.widget_key(1, ("lessons", 0, "quiz", 0, "explanation"))] = "Teacher-edited explanation."
    evaluation = ui.evaluate_course(ui.collect_edits(state, 1, data), sel, doc.page_count)
    fp = ui.fingerprint(evaluation.course) if approve else None
    return ui.with_review_status(evaluation.course, fp), evaluation.report


def test_json_export_reflects_latest_edits_and_status(sample):
    course, report = edited_course(sample)
    data = json.loads(ui.course_to_json(course, report))
    lesson = data["course"]["lessons"][0]
    assert lesson["title"] == "Teacher-edited title"
    assert lesson["quiz"][0]["explanation"] == "Teacher-edited explanation."
    meta = data["course"]["metadata"]
    assert (meta["generation_mode"], meta["review_status"]) == ("demo", "draft")
    assert (meta["start_page"], meta["end_page"]) == (2, 5)
    assert data["citation_check"]["note"] == TRACEABILITY_NOTE
    first = data["citation_check"]["results"][0]
    assert first["status"] == "verified" and first["page"] >= 2 and first["quote"]


def test_markdown_export_reflects_latest_edits_and_status(sample):
    course, report = edited_course(sample)
    md = ui.course_to_markdown(course, report)
    assert "## Lesson 1: Teacher-edited title" in md
    assert "*Explanation:* Teacher-edited explanation." in md
    assert "Draft (not yet approved)" in md and "Approved by the teacher" not in md
    assert "Offline demo" in md and "no AI used" in md
    assert "(correct)" in md and "*Answer:*" in md
    assert "> PDF page 2: “" in md and "(Verified)" in md
    assert TRACEABILITY_NOTE in md


def test_exports_show_approval_only_after_approval(sample):
    course, report = edited_course(sample, approve=True)
    assert json.loads(ui.course_to_json(course, report))["course"]["metadata"]["review_status"] == "approved"
    assert "Approved by the teacher" in ui.course_to_markdown(course, report)


# ---------------------------------------------------------------- generation outcomes

def test_demo_generation_succeeds_without_an_api_key():
    doc, sel = chapter_selection()
    outcome = ui.run_generation(sel, doc, "demo")
    assert outcome.ok and outcome.result.course.metadata.generation_mode == "demo"
    assert ui.generation_info(outcome.result)["mode"] == "demo"


@pytest.mark.parametrize("error,title", [
    (api_error(400, "INVALID_ARGUMENT", "API key not valid.", KEY_DETAILS), "The API key was rejected"),
    (api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", QUOTA_DETAILS), "Gemini free-tier limit reached"),
    (api_error(503, "UNAVAILABLE", "The model is overloaded."), "The AI request failed"),
])
def test_ai_failures_become_outcomes_that_offer_demo_mode(error, title):
    doc, sel = chapter_selection()
    outcome = ui.run_generation(sel, doc, "llm", config=CONFIG, client=FakeClient(error))
    assert not outcome.ok and outcome.title == title and outcome.offer_demo
    assert outcome.result is None  # no demo course is substituted
    assert SECRET not in outcome.message


def test_invalid_ai_output_lists_problems():
    four = json.dumps({"lessons": json.loads(course_json())["lessons"][:4]})
    doc, sel = chapter_selection()
    outcome = ui.run_generation(sel, doc, "llm", config=CONFIG, client=FakeClient(four, four, four))
    assert outcome.title == "The AI output could not be accepted" and outcome.details


def test_unknown_model_is_reported_as_unavailable():
    doc, sel = chapter_selection()
    client = FakeClient(model_error=api_error(404, "NOT_FOUND", "model not found"))
    outcome = ui.run_generation(sel, doc, "llm", config=CONFIG, client=client)
    assert outcome.title == "AI is not available" and "gemini-3.8-flash" in outcome.message


def test_auto_mode_without_ai_offers_demo_instead_of_using_it(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    doc, sel = chapter_selection()
    outcome = ui.run_generation(sel, doc, "auto")
    assert not outcome.ok and outcome.result is None
    assert outcome.title == "AI is not available" and outcome.offer_demo


def test_insufficient_text_does_not_offer_demo_mode():
    doc, sel = chapter_selection([None, None])
    outcome = ui.run_generation(sel, doc, "demo")
    assert outcome.title == "Not enough usable text" and not outcome.offer_demo


def test_successful_ai_generation_with_a_mocked_client():
    doc, sel = chapter_selection()
    outcome = ui.run_generation(sel, doc, "llm", config=CONFIG, client=FakeClient(course_json()))
    assert outcome.ok and ui.generation_info(outcome.result)["model"] == "gemini-3.8-flash"


# ---------------------------------------------------------------- AI status

def test_ai_status_never_shows_the_key(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.delenv("COURSEFORGE_MODEL", raising=False)
    ready, message = ui.ai_status()
    assert ready and "gemini-3.8-flash" in message and "free tier" in message and SECRET not in message


def test_free_tier_notice_states_quotas_and_data_use():
    assert "Google sets the quotas" in ui.FREE_TIER_NOTICE
    assert "may be used by Google to improve its products" in ui.FREE_TIER_NOTICE
    assert "without billing" in ui.FREE_TIER_NOTICE


def test_ai_status_when_not_configured(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    ready, message = ui.ai_status()
    assert not ready and "LLM_PROVIDER is not set" in message
