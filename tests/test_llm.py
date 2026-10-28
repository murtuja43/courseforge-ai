"""Tests for AI course generation. Every API response is mocked; no request leaves the machine."""

import json
import socket
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

import courseforge.llm as llm
from courseforge.generator import DemoResult, InsufficientSourceError
from courseforge.llm import (
    LLMAuthError, LLMConfig, LLMConfigError, LLMConnectionError, LLMOutputError, LLMRateLimitError,
    LLMRequestError, LLMResult, LLMServiceError, LLMTimeoutError, generate_course,
    generate_llm_course, load_config,
)
from courseforge.validate import CitationStatus
from test_generator import FICTIONAL_CHAPTER, chapter_selection

SECRET = "sk-test-not-a-real-key-0123456789"
CONFIG = LLMConfig(provider="anthropic", model="test-model", api_key=SECRET)

# Exact sentences from the fictional fixture chapter, by physical page.
QUOTES = {
    1: "The moss produces a soft blue light at night by storing sunlight during the day.",
    2: "Glowmoss spreads by releasing silver spores into the cliff winds.",
    3: "The glow of the moss attracts night insects called dimwings.",
    4: "Cold winters reduce the amount of stored glow energy in each colony.",
}
LESSON_PAGES = [1, 2, 2, 3, 4]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("a real network connection was attempted")
    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


# ---------------------------------------------------------------- fakes

def ref(page, quote=None):
    return {"page": page, "quote": quote or QUOTES[page]}


def lesson(n, page, quote=None, example_sources=None):
    return {
        "number": n,
        "title": f"Glowmoss topic {n}",
        "objectives": [{"text": f"Describe idea {n} about glowmoss."}],
        "explanation": "A student-friendly explanation grounded in the cited page.",
        "examples": [{"title": "Example", "explanation": "An example from the source.",
                      "sources": [ref(page, quote)] if example_sources is None else example_sources}],
        "practice": [{"prompt": "Explain the idea.", "suggested_answer": "A short answer.",
                      "sources": [ref(page, quote)]}],
        "quiz": [{"question": f"Question {n}?",
                  "options": [{"id": i, "text": f"Option {i}"} for i in "ABCD"],
                  "correct_option": "B", "explanation": "B is supported by the source.",
                  "sources": [ref(page, quote)]}],
        "sources": [ref(page, quote)],
    }


def course_json(**changes):
    lessons = [lesson(n, p) for n, p in enumerate(LESSON_PAGES, start=1)]
    for index, replacement in changes.items():
        lessons[int(index.removeprefix("lesson"))] = replacement
    return json.dumps({"lessons": lessons})


def message(text, stop_reason="end_turn"):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=1000, output_tokens=500))


class FakeStream:
    def __init__(self, outcome):
        self.outcome = outcome

    def __enter__(self):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.outcome


class FakeClient:
    """Plays back scripted responses and records every request."""

    def __init__(self, *outcomes, model_error=None):
        self.outcomes = list(outcomes)
        self.requests = []
        self.model_error = model_error
        self.messages = SimpleNamespace(stream=self._stream)
        self.models = SimpleNamespace(retrieve=self._retrieve)

    def _retrieve(self, model):
        if self.model_error:
            raise self.model_error
        return SimpleNamespace(id=model)

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        return FakeStream(message(outcome) if isinstance(outcome, str) else outcome)


def api_error(cls, status, headers=None):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request, headers=headers or {})
    return cls("error from the service", response=response, body=None)


def run(client, **kwargs):
    doc, sel = chapter_selection()
    return generate_llm_course(sel, doc.filename, config=CONFIG, client=client,
                               document_page_count=doc.page_count, **kwargs)


# ---------------------------------------------------------------- valid responses

def test_valid_response_gives_a_verified_ai_course():
    client = FakeClient(course_json())
    result = run(client)
    assert isinstance(result, LLMResult)
    assert result.verified and result.attempts == 1 and result.problems == []
    meta = result.course.metadata
    assert (meta.generation_mode, meta.model, meta.review_status) == ("llm", "test-model", "draft")
    assert all(c.status == CitationStatus.VERIFIED for c in result.citations.checks)
    assert result.usage == {"input_tokens": 1000, "output_tokens": 500}
    assert "Generated by the AI model test-model" in result.status_message
    assert "do not confirm that explanations" in result.status_message


def test_request_contains_page_tagged_source_and_structured_output_schema():
    client = FakeClient(course_json())
    run(client)
    [request] = client.requests
    assert request["model"] == "test-model"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "effort" not in request["output_config"]
    source_block = request["messages"][0]["content"][0]
    assert source_block["cache_control"] == {"type": "ephemeral"}
    for n in (1, 2, 3, 4):
        assert f'<page number="{n}">' in source_block["text"]
    assert "Cite only pages" in request["system"]


def test_effort_is_sent_only_when_configured():
    client = FakeClient(course_json())
    doc, sel = chapter_selection()
    config = LLMConfig("anthropic", "test-model", SECRET, effort="high")
    generate_llm_course(sel, doc.filename, config=config, client=client)
    assert client.requests[0]["output_config"]["effort"] == "high"


def test_api_key_never_appears_in_requests_or_results():
    client = FakeClient(course_json())
    result = run(client)
    assert SECRET not in repr(CONFIG)
    assert SECRET not in json.dumps(client.requests, default=str)
    assert SECRET not in result.course.model_dump_json() + result.status_message


def test_scanned_pages_are_not_sent_to_the_model():
    client = FakeClient(course_json())
    doc, sel = chapter_selection(FICTIONAL_CHAPTER + [None], scanned_pages={5})
    generate_llm_course(sel, doc.filename, config=CONFIG, client=client)
    assert '<page number="5">' not in client.requests[0]["messages"][0]["content"][0]["text"]


# ---------------------------------------------------------------- prompt content (fix E)

def test_request_lists_the_selected_sections_with_physical_start_pages():
    from courseforge.pdf_extract import select_page_range
    from test_pdf_extract import section_doc
    doc = section_doc()
    sel = select_page_range(doc, 2, 3)
    text = llm.build_section_context(sel)
    assert "- CHAPTER 4: Crystals of Orvane (began on page 1, before the selected pages; continues here)" in text
    assert "  - 4.2 Crystal Growth (began on page 1, before the selected pages; continues here; " \
           "ends partway down page 3)" in text
    assert "  - 4.3 Crystal Harvesting (starts on page 3)" in text


def test_selection_without_sections_says_so():
    _, sel = chapter_selection()
    assert "no table of contents" in llm.build_section_context(sel)


def test_page_tags_carry_printed_labels_separately():
    from courseforge.pdf_extract import select_page_range
    from test_pdf_extract import section_doc
    source = llm.build_source_text(select_page_range(section_doc(), 2, 3))
    assert '<page number="2" printed="42">' in source and '<page number="3" printed="43">' in source


def test_section_context_is_sent_with_the_source_text():
    from courseforge.pdf_extract import select_page_range
    from test_pdf_extract import section_doc
    doc = section_doc()
    client = FakeClient("{not json", "{not json", "{not json")
    with pytest.raises(LLMOutputError):
        generate_llm_course(select_page_range(doc, 2, 3), "fiction.pdf", config=CONFIG, client=client)
    text = client.requests[0]["messages"][0]["content"][0]["text"]
    assert text.index("Textbook sections overlapping") < text.index('<page number="2" printed="42">')


@pytest.mark.parametrize("rule", [
    "Teach, do not copy",
    "Do not string together sentences copied from the textbook",
    "organize the lessons around it",
    "never mixes unrelated sections",
    "Equations are essential; teach them",
    "Never present your own reconstruction of a garbled equation",
    'add a sentence beginning "Teacher check:"',
    "Never use the printed page number",
    "Never invent a quotation",
    "Never state that a quotation or source has been verified",
    "check the arithmetic so exactly one option is correct",
])
def test_system_prompt_states_the_rule(rule):
    assert rule in llm.SYSTEM_PROMPT


def test_system_prompt_does_not_tell_the_model_to_skip_mathematics():
    prompt = llm.SYSTEM_PROMPT.casefold()
    for phrase in ("ignore equations", "avoid equations", "do not teach math", "avoid mathematics"):
        assert phrase not in prompt


def test_teacher_check_flags_become_a_notice():
    flagged = json.loads(course_json())
    flagged["lessons"][1]["explanation"] += " Teacher check: Eq. (9.36) is garbled in the extracted text."
    flagged["lessons"][3]["examples"][0]["explanation"] += " Teacher check: steps 3-4 were worked out for this course."
    result = run(FakeClient(json.dumps(flagged)))
    assert result.verified
    assert llm.teacher_check_locations(result.course) == ["Lesson 2", "Lesson 4, example 1"]
    assert any("flagged 2 item(s) for teacher checking" in n for n in result.notices)


def test_no_teacher_check_means_no_notice():
    assert not any("teacher checking" in n for n in run(FakeClient(course_json())).notices)


# ---------------------------------------------------------------- invalid output and retries

def test_invalid_json_is_retried_with_feedback():
    client = FakeClient("{not json", course_json())
    result = run(client)
    assert result.verified and result.attempts == 2
    retry_messages = client.requests[1]["messages"]
    assert retry_messages[1] == {"role": "assistant", "content": "{not json"}
    assert "not valid JSON" in retry_messages[2]["content"]


def test_schema_errors_are_fed_back():
    bad = json.loads(course_json())
    bad["lessons"][2]["quiz"][0]["correct_option"] = "E"
    client = FakeClient(json.dumps(bad), course_json())
    result = run(client)
    assert result.verified and result.attempts == 2
    assert "lessons 3 > quiz 1: correct_option 'E'" in client.requests[1]["messages"][2]["content"]


def test_persistently_invalid_output_stops_after_the_attempt_limit():
    four_lessons = json.dumps({"lessons": json.loads(course_json())["lessons"][:4]})
    client = FakeClient(four_lessons, four_lessons, four_lessons, four_lessons)
    with pytest.raises(LLMOutputError, match="after 3 attempt") as info:
        run(client)
    assert len(client.requests) == 3  # bounded cost: never more than MAX_ATTEMPTS calls
    assert info.value.problems and info.value.problems[0].startswith("lessons:")


def test_unexpected_top_level_fields_are_rejected():
    data = json.loads(course_json())
    data["metadata"] = {"title": "model-made metadata"}
    client = FakeClient(json.dumps(data), course_json())
    assert run(client).attempts == 2
    assert 'only a "lessons" list' in client.requests[1]["messages"][2]["content"]


def test_empty_response_is_retried_without_an_empty_assistant_turn():
    client = FakeClient("", course_json())
    assert run(client).attempts == 2
    roles = [m["role"] for m in client.requests[1]["messages"]]
    assert roles == ["user", "user"]


# ---------------------------------------------------------------- citation failures

def test_citation_failure_gets_a_corrective_retry():
    wrong = lesson(2, 2, quote="Glowmoss spreads by releasing golden seeds into the sea.")
    client = FakeClient(course_json(lesson1=wrong), course_json())
    result = run(client)
    assert result.verified and result.attempts == 2
    feedback = client.requests[1]["messages"][2]["content"]
    assert "Lesson 2: quote not found on page 2" in feedback


def test_citation_to_a_page_outside_the_selection_is_reported():
    outside = lesson(1, 1)
    outside["sources"] = [{"page": 9, "quote": QUOTES[1]}]
    client = FakeClient(course_json(lesson0=outside), course_json())
    run(client)
    assert "page 9 does not exist" in client.requests[1]["messages"][2]["content"]


def test_uncited_example_is_a_problem_for_ai_output():
    uncited = lesson(1, 1, example_sources=[])
    client = FakeClient(course_json(lesson0=uncited), course_json())
    run(client)
    assert "Lesson 1, example 1: has no source reference" in client.requests[1]["messages"][2]["content"]


def test_persistent_citation_failure_returns_an_explicitly_unverified_draft():
    wrong = course_json(lesson4=lesson(5, 4, quote="Warm summers make the moss glow brighter."))
    client = FakeClient(wrong, wrong, wrong)
    result = run(client)
    assert len(client.requests) == 3
    assert not result.verified
    assert result.attempts == 3
    assert result.problems == ["Lesson 5: quote not found on page 4: \"Warm summers make the moss glow brighter.\"",
                               "Lesson 5, example 1: quote not found on page 4: \"Warm summers make the moss glow brighter.\"",
                               "Lesson 5, practice activity 1: quote not found on page 4: \"Warm summers make the moss glow brighter.\"",
                               "Lesson 5, quiz question 1: quote not found on page 4: \"Warm summers make the moss glow brighter.\""]
    bad = [c for c in result.citations.checks if c.location.startswith("Lesson 5")]
    assert bad and not any(c.verified for c in bad)
    assert result.status_message.count("UNVERIFIED DRAFT") == 1


# ---------------------------------------------------------------- configuration

def test_missing_provider_means_ai_is_off():
    with pytest.raises(LLMConfigError, match="LLM_PROVIDER is not set"):
        load_config({})


@pytest.mark.parametrize("env,message", [
    ({"LLM_PROVIDER": "anthropic", "COURSEFORGE_MODEL": "m"}, "ANTHROPIC_API_KEY is not set"),
    ({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": SECRET}, "COURSEFORGE_MODEL is not set"),
    ({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "  ", "COURSEFORGE_MODEL": "m"}, "ANTHROPIC_API_KEY"),
    ({"LLM_PROVIDER": "magic", "ANTHROPIC_API_KEY": SECRET, "COURSEFORGE_MODEL": "m"}, "not supported"),
    ({"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": SECRET, "COURSEFORGE_MODEL": "m",
      "COURSEFORGE_EFFORT": "extreme"}, "COURSEFORGE_EFFORT 'extreme' is not valid"),
])
def test_incomplete_configuration_is_reported(env, message):
    with pytest.raises(LLMConfigError, match=message) as info:
        load_config(env)
    assert SECRET not in str(info.value)


def test_complete_configuration_is_loaded_without_exposing_the_key():
    config = load_config({"LLM_PROVIDER": " Anthropic ", "ANTHROPIC_API_KEY": SECRET,
                          "COURSEFORGE_MODEL": "some-model", "COURSEFORGE_EFFORT": "Medium"})
    assert (config.provider, config.model, config.effort) == ("anthropic", "some-model", "medium")
    assert config.api_key == SECRET and SECRET not in repr(config)


def test_unknown_model_is_rejected_before_any_generation():
    client = FakeClient(course_json(), model_error=api_error(anthropic.NotFoundError, 404))
    with pytest.raises(LLMConfigError, match="model 'test-model' was not found"):
        run(client)
    assert client.requests == []


# ---------------------------------------------------------------- API errors

@pytest.mark.parametrize("error,expected,message", [
    (api_error(anthropic.AuthenticationError, 401), LLMAuthError, "rejected the API key"),
    (api_error(anthropic.PermissionDeniedError, 403), LLMAuthError, "rejected the API key"),
    (api_error(anthropic.RateLimitError, 429, {"retry-after": "30"}), LLMRateLimitError, "about 30 seconds"),
    (api_error(anthropic.BadRequestError, 400), LLMRequestError, "rejected the request"),
    (api_error(anthropic.InternalServerError, 500), LLMServiceError, "HTTP 500"),
    (anthropic.APITimeoutError(request=httpx2.Request("POST", "https://api.anthropic.com")),
     LLMTimeoutError, "did not respond in time"),
    (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")),
     LLMConnectionError, "Could not connect"),
])
def test_api_errors_become_clear_messages(error, expected, message):
    client = FakeClient(error)
    with pytest.raises(expected, match=message) as info:
        run(client)
    assert SECRET not in str(info.value)
    assert len(client.requests) == 1  # our code adds no retries on top of the SDK's own


def test_rate_limit_keeps_the_retry_after_value():
    client = FakeClient(api_error(anthropic.RateLimitError, 429, {"retry-after": "30"}))
    with pytest.raises(LLMRateLimitError) as info:
        run(client)
    assert info.value.retry_after == "30"


def test_authentication_error_while_checking_the_model():
    client = FakeClient(model_error=api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(LLMAuthError):
        run(client)


@pytest.mark.parametrize("stop_reason,expected", [("refusal", "declined"), ("max_tokens", "cut off")])
def test_refusal_and_truncation_are_not_retried(stop_reason, expected):
    client = FakeClient(message(course_json(), stop_reason=stop_reason))
    with pytest.raises(LLMOutputError, match=expected):
        run(client)
    assert len(client.requests) == 1


# ---------------------------------------------------------------- input limits

def test_oversized_source_is_rejected_before_calling_the_api(monkeypatch):
    monkeypatch.setattr(llm, "MAX_SOURCE_CHARS", 100)
    client = FakeClient(course_json())
    with pytest.raises(LLMRequestError, match="limit is 100"):
        run(client)
    assert client.requests == []


def test_chapter_without_text_is_rejected_before_calling_the_api():
    client = FakeClient(course_json())
    doc, sel = chapter_selection([None, None])
    with pytest.raises(InsufficientSourceError):
        generate_llm_course(sel, doc.filename, config=CONFIG, client=client)
    assert client.requests == []


# ---------------------------------------------------------------- offline fallback

@pytest.fixture
def no_ai_environment(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: False)  # ignore any local .env
    for name in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "COURSEFORGE_MODEL", "COURSEFORGE_EFFORT"):
        monkeypatch.delenv(name, raising=False)


def test_auto_mode_falls_back_to_demo_when_ai_is_not_configured(no_ai_environment):
    doc, sel = chapter_selection()
    result = generate_course(sel, doc.filename, mode="auto")
    assert isinstance(result, DemoResult)
    assert result.course.metadata.generation_mode == "demo"
    assert result.notices[0] == ("AI generation is off because LLM_PROVIDER is not set. "
                                 "The offline demo generator was used instead.")


def test_llm_mode_without_configuration_raises(no_ai_environment):
    doc, sel = chapter_selection()
    with pytest.raises(LLMConfigError):
        generate_course(sel, doc.filename, mode="llm")


def test_demo_mode_never_calls_the_api():
    client = FakeClient()
    doc, sel = chapter_selection()
    result = generate_course(sel, doc.filename, mode="demo", config=CONFIG, client=client)
    assert isinstance(result, DemoResult) and client.requests == []


def test_auto_mode_uses_ai_when_configured():
    client = FakeClient(course_json())
    doc, sel = chapter_selection()
    result = generate_course(sel, doc.filename, mode="auto", config=CONFIG, client=client)
    assert isinstance(result, LLMResult) and result.verified


def test_real_client_construction_makes_no_request():
    client = llm.make_client(CONFIG)  # constructing the SDK client does not connect
    assert isinstance(client, anthropic.Anthropic)
