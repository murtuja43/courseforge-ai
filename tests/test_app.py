"""End-to-end tests of app.py using Streamlit's AppTest. AI responses are mocked."""

import json
import socket
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import courseforge.llm as llm
from courseforge import ui_helpers as ui
from courseforge.sample import CHAPTER_1, sample_pdf_bytes
from test_llm import QUOTA_DETAILS, FakeClient, api_error, lesson

APP = str(Path(__file__).resolve().parent.parent / "app.py")
AI_VARS = ("LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY", "COURSEFORGE_MODEL")
FAKE_KEY = "AIza-test-not-a-real-key"


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", None, ""}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """No outside network, no local .env, and AI unconfigured unless a test configures it.

    AppTest's event loop needs local loopback sockets, so instead of blocking sockets
    outright, any lookup of a non-local host fails. A real Gemini call must resolve
    generativelanguage.googleapis.com first, so it cannot get through.
    """
    real_getaddrinfo = socket.getaddrinfo

    def local_only(host, *args, **kwargs):
        if host not in LOCAL_HOSTS:
            raise AssertionError(f"a real network connection to {host} was attempted")
        return real_getaddrinfo(host, *args, **kwargs)
    monkeypatch.setattr(socket, "getaddrinfo", local_only)
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: False)
    for name in AI_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def configured_ai(monkeypatch):
    """Configure Gemini with a fake key and route the client to a scripted fake."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    holder = {}
    monkeypatch.setattr(llm, "make_client", lambda config: holder["client"])
    return holder


def start():
    return AppTest.from_file(APP, default_timeout=60).run()


def with_demo_course():
    at = start()
    at.button(key="load_sample").click().run()
    at.radio(key="mode_choice").set_value("demo").run()
    at.button(key="generate").click().run()
    return at


def key(path, course_id=1):
    return ui.widget_key(course_id, path)


def badges(at):
    return " ".join(m.value for m in at.markdown if 'class="cf-badge' in m.value)


@pytest.mark.parametrize("host", ["generativelanguage.googleapis.com", "api.anthropic.com"])
def test_network_guard_blocks_hosted_ai_services(host):
    with pytest.raises(AssertionError, match=host):
        socket.getaddrinfo(host, 443)


# ---------------------------------------------------------------- start and upload

def test_app_starts_with_the_landing_page():
    at = start()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "CourseForge AI" in page and "PROJECT AXIORA" in page
    assert "From Textbooks to Structured Learning Experiences" in page
    assert "Teachers remain responsible" in page
    assert any("Upload a textbook PDF" in i.value for i in at.info)


def test_uploaded_pdf_is_extracted_and_its_chapters_listed():
    at = start()
    at.get("file_uploader")[0].upload("my-book.pdf", sample_pdf_bytes(), "application/pdf").run()
    assert not at.exception
    assert [m.value for m in at.metric][:3] == ["8", "8", "2"]
    assert at.radio(key=f"method|{at.session_state.doc_id}").value == "toc"
    chapter = at.selectbox(key=f"chapter|{at.session_state.doc_id}")
    chapter.set_value(1).run()
    assert any("Chapter 2: Dimwings and the Night Cliffs, physical PDF pages 6-8" in m.value for m in at.markdown)


def test_invalid_upload_shows_an_error_without_crashing():
    at = start()
    at.get("file_uploader")[0].upload("notes.pdf", b"not really a pdf", "application/pdf").run()
    assert not at.exception
    assert any("This PDF could not be used: This file is not a PDF." in e.value for e in at.error)


def test_page_range_selection_is_validated():
    at = start()
    at.button(key="load_sample").click().run()
    doc_id = at.session_state.doc_id
    at.radio(key=f"method|{doc_id}").set_value("range").run()
    at.number_input(key=f"start|{doc_id}").set_value(6).run()
    at.number_input(key=f"end|{doc_id}").set_value(3).run()
    assert any("Start page must not be after the end page" in e.value for e in at.error)
    assert at.button(key="generate").disabled


# ---------------------------------------------------------------- demo generation

def test_demo_mode_generates_a_course_without_an_api_key():
    at = with_demo_course()
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == [f"Lesson {n}" for n in range(1, 6)] + ["Citations", "Approve & export"]
    assert "Offline demo" in badges(at) and "Draft" in badges(at)
    assert "41 of 41 citations verified" in badges(at)
    assert at.session_state.gen_info["mode"] == "demo"


def test_auto_mode_without_ai_explains_and_does_not_build_a_demo_course():
    at = start()
    at.button(key="load_sample").click().run()
    assert any("AI is unavailable" in w.value for w in at.warning)
    at.button(key="generate").click().run()  # "auto" is the default mode
    assert at.session_state.course_data is None
    assert any("AI is not available" in e.value and "LLM_PROVIDER is not set" in e.value for e in at.error)
    at.button(key="switch_demo").click().run()  # demo only when explicitly chosen
    assert at.session_state.gen_info["mode"] == "demo"


def test_second_click_does_not_replace_the_course_without_confirmation():
    at = with_demo_course()
    at.text_input(key=key(("lessons", 0, "title"))).input("My edited title").run()
    at.button(key="generate").click().run()
    assert at.session_state.course_id == 1
    assert at.text_input(key=key(("lessons", 0, "title"))).value == "My edited title"
    assert any("A course already exists" in i.value for i in at.info)


# ---------------------------------------------------------------- editing

def test_edits_persist_across_lessons_and_reruns():
    at = with_demo_course()
    at.text_input(key=key(("lessons", 0, "title"))).input("Edited lesson one").run()
    at.text_area(key=key(("lessons", 3, "explanation"))).input("Edited lesson four explanation.").run()
    at.selectbox(key=key(("lessons", 2, "quiz", 0, "correct_option"))).set_value("A").run()
    at.run()
    lessons = at.session_state.course_data["lessons"]
    assert lessons[0]["title"] == "Edited lesson one"
    assert lessons[3]["explanation"] == "Edited lesson four explanation."
    assert lessons[2]["quiz"][0]["correct_option"] == "A"
    assert at.text_input(key=key(("lessons", 0, "title"))).value == "Edited lesson one"


def test_blank_edit_pauses_approval_and_export():
    at = with_demo_course()
    at.text_input(key=key(("lessons", 1, "title"))).input("").run()
    assert any("Some edits are not valid yet" in e.value for e in at.error)
    assert not at.get("download_button")
    assert not any(b.key == "approve" for b in at.button)


# ---------------------------------------------------------------- citations

def test_broken_quote_is_shown_as_not_verified():
    at = with_demo_course()
    at.text_area(key=key(("lessons", 0, "sources", 0, "quote"))).input("Glowmoss is made of silver.").run()
    assert "40 of 41 citations verified" in badges(at)
    assert 'cf-warn' in badges(at)
    rows = at.dataframe[0].value
    assert "Not verified: quote not found on the page" in rows["Status"].tolist()
    assert any("quote not found on page" in c.value for c in at.caption)


def test_citation_outside_the_chapter_is_shown():
    at = with_demo_course()
    at.number_input(key=key(("lessons", 0, "sources", 0, "page"))).set_value(7).run()
    assert "Not verified: outside the selected chapter" in at.dataframe[0].value["Status"].tolist()


# ---------------------------------------------------------------- approval

def test_approval_requires_confirmation_and_is_revoked_by_edits():
    at = with_demo_course()
    assert at.button(key="approve").disabled
    at.checkbox(key="reviewed|1").check().run()
    at.button(key="approve").click().run()
    assert "Approved by teacher" in badges(at)
    at.text_input(key=key(("lessons", 0, "title"))).input("A late edit").run()
    assert "Draft" in badges(at) and "Approved by teacher" not in badges(at)
    assert any("edited after approval" in i.value for i in at.info)


def test_unverified_citations_need_an_extra_acknowledgement():
    at = with_demo_course()
    at.text_area(key=key(("lessons", 0, "sources", 0, "quote"))).input("Glowmoss is made of silver.").run()
    at.checkbox(key="reviewed|1").check().run()
    assert at.button(key="approve").disabled
    at.checkbox(key="ack|1").check().run()
    assert not at.button(key="approve").disabled


# ---------------------------------------------------------------- AI mode

def test_ai_mode_without_configuration_offers_demo_but_does_not_switch():
    at = start()
    at.button(key="load_sample").click().run()
    at.radio(key="mode_choice").set_value("llm").run()
    at.button(key="generate").click().run()
    assert at.session_state.course_data is None
    assert any("AI is not available" in e.value for e in at.error)
    at.button(key="switch_demo").click().run()
    assert at.session_state.gen_info["mode"] == "demo"
    assert at.radio(key="mode_choice").value == "demo"


def test_gemini_quota_error_is_reported_and_no_demo_course_is_substituted(configured_ai):
    client = FakeClient(api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", QUOTA_DETAILS))
    configured_ai["client"] = client
    at = start()
    at.button(key="load_sample").click().run()
    at.radio(key="mode_choice").set_value("auto").run()
    at.button(key="generate").click().run()
    assert not at.exception
    assert at.session_state.course_data is None
    assert any("Gemini free-tier limit reached" in e.value and "37 seconds" in e.value
               and "switch to another provider" in e.value for e in at.error)
    assert len(client.requests) == 1
    assert at.button(key="switch_demo")


def test_free_tier_notice_is_shown_when_gemini_is_configured(configured_ai):
    configured_ai["client"] = FakeClient()
    at = start()
    at.button(key="load_sample").click().run()
    notices = " ".join(c.value for c in at.caption)
    assert "may be used by Google to improve its products" in notices
    assert any("gemini-3.8-flash" in s.value for s in at.success)
    page = " ".join(m.value for m in at.markdown) + notices + " ".join(s.value for s in at.success)
    assert FAKE_KEY not in page


def test_successful_ai_generation_is_labelled_as_ai(configured_ai):
    # Sample chapter 1 is on PDF pages 2-5; quote one real sentence from each page.
    quotes = {p: CHAPTER_1[p - 2].split("\n")[-1].split(". ")[0] + "." for p in (2, 3, 4, 5)}
    lessons = [lesson(n, p, quote=quotes[p]) for n, p in enumerate([2, 3, 3, 4, 5], start=1)]
    configured_ai["client"] = FakeClient(json.dumps({"lessons": lessons}))
    at = start()
    at.button(key="load_sample").click().run()
    at.radio(key="mode_choice").set_value("llm").run()
    at.button(key="generate").click().run()
    assert not at.exception and not at.error
    assert "AI-generated" in badges(at) and "gemini-3.8-flash" in badges(at)
    assert "citations verified" in badges(at) and "cf-ok" in badges(at)
