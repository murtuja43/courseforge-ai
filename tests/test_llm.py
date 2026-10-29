"""Tests for AI course generation with the Gemini free tier.

Every Gemini response is mocked with the SDK's own response types; no request leaves
the machine.
"""

import json
import socket

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

import courseforge.llm as llm
from courseforge.gemini import GeminiProvider, gemini_schema
from courseforge.generator import DemoResult, InsufficientSourceError
from courseforge.llm import (
    LLMAuthError, LLMConfig, LLMConfigError, LLMConnectionError, LLMOutputError, LLMRateLimitError,
    LLMRequestError, LLMResult, LLMServiceError, LLMTimeoutError, generate_course,
    generate_llm_course, load_config,
)
from courseforge.validate import CitationStatus
from test_generator import FICTIONAL_CHAPTER, chapter_selection

SECRET = "AIza-test-not-a-real-key-0123456789"
CONFIG = LLMConfig(provider="gemini", model="gemini-3.8-flash", api_key=SECRET)

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


def gemini_response(text, finish="STOP", thought=None):
    """A real google.genai response object, as generate_content returns it."""
    parts = ([types.Part(text=thought, thought=True)] if thought else []) + [types.Part(text=text)]
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts),
                                    finish_reason=types.FinishReason(finish))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1000, candidates_token_count=500, thoughts_token_count=200))


def blocked_response(reason="SAFETY"):
    return types.GenerateContentResponse(
        candidates=[], prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason(reason)))


class FakeModels:
    def __init__(self, client):
        self.client = client

    def get(self, *, model, config=None):
        self.client.model_checks.append(model)
        if self.client.model_error:
            raise self.client.model_error
        return types.Model(name=f"models/{model}")

    def generate_content(self, *, model, contents, config=None):
        self.client.requests.append({"model": model, "contents": contents, "config": config})
        outcome = self.client.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return gemini_response(outcome) if isinstance(outcome, str) else outcome


class FakeClient:
    """Plays back scripted Gemini outcomes and records every request."""

    def __init__(self, *outcomes, model_error=None):
        self.outcomes = list(outcomes)
        self.requests = []
        self.model_checks = []
        self.model_error = model_error
        self.models = FakeModels(self)


def api_error(code, status="", message="error from the service", details=None):
    """A real google-genai APIError, shaped like a Gemini error response."""
    body = {"error": {"code": code, "message": message, "status": status, "details": details or []}}
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, body)


def turn_text(request, index):
    return request["contents"][index].parts[0].text


def roles(request):
    return [c.role for c in request["contents"]]


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
    assert (meta.generation_mode, meta.model, meta.review_status) == ("llm", "gemini-3.8-flash", "draft")
    assert all(c.status == CitationStatus.VERIFIED for c in result.citations.checks)
    assert result.usage == {"input_tokens": 1000, "output_tokens": 500, "thinking_tokens": 200}
    assert "Generated by the AI model gemini-3.8-flash" in result.status_message
    assert "do not confirm that explanations" in result.status_message


def test_request_uses_structured_json_output_and_the_system_prompt():
    client = FakeClient(course_json())
    run(client)
    [request] = client.requests
    config = request["config"]
    assert request["model"] == "gemini-3.8-flash"
    assert config.system_instruction == llm.SYSTEM_PROMPT
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == gemini_schema(llm.CourseDraft)
    assert config.response_schema is None  # only one schema form may be sent
    assert config.max_output_tokens == llm.MAX_OUTPUT_TOKENS
    assert config.candidate_count == 1
    assert config.temperature is None  # Gemini 3.6+ ignores sampling settings
    assert roles(request) == ["user"]
    for n in (1, 2, 3, 4):
        assert f'<page number="{n}">' in turn_text(request, 0)


def test_model_is_checked_once_before_generating():
    client = FakeClient(course_json())
    run(client)
    assert client.model_checks == ["gemini-3.8-flash"]


def test_thought_parts_are_not_treated_as_the_answer():
    client = FakeClient(gemini_response(course_json(), thought="planning the lessons..."))
    assert run(client).verified


def test_api_key_never_appears_in_requests_or_results():
    client = FakeClient(course_json())
    result = run(client)
    assert SECRET not in repr(CONFIG)
    assert SECRET not in repr(client.requests)
    assert SECRET not in result.course.model_dump_json() + result.status_message + repr(result.usage)


def test_scanned_pages_are_not_sent_to_the_model():
    client = FakeClient(course_json())
    doc, sel = chapter_selection(FICTIONAL_CHAPTER + [None], scanned_pages={5})
    generate_llm_course(sel, doc.filename, config=CONFIG, client=client)
    assert '<page number="5">' not in turn_text(client.requests[0], 0)


# ---------------------------------------------------------------- Gemini schema

def test_gemini_schema_uses_only_supported_keywords():
    from courseforge.gemini import _SUPPORTED_KEYWORDS
    schema = gemini_schema(llm.CourseDraft)
    seen = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("properties", "$defs"):
                    for sub in value.values():
                        walk(sub)
                else:
                    seen.add(key)
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    assert seen <= _SUPPORTED_KEYWORDS
    assert not {"pattern", "minLength", "maxLength", "default"} & seen


def test_gemini_schema_requires_sources_and_forbids_extra_fields():
    defs = gemini_schema(llm.CourseDraft)["$defs"]
    for name in ("Example", "PracticeActivity", "QuizQuestion", "Lesson"):
        assert "sources" in defs[name]["required"]
        assert defs[name]["additionalProperties"] is False


def test_ref_nodes_carry_nothing_else():
    def refs(node):
        if isinstance(node, dict):
            if "$ref" in node:
                yield node
            for value in node.values():
                yield from refs(value)
        elif isinstance(node, list):
            for item in node:
                yield from refs(item)
    assert all(set(n) == {"$ref"} for n in refs(gemini_schema(llm.CourseDraft)))


# ---------------------------------------------------------------- prompt content

def test_request_lists_the_selected_sections_with_physical_start_pages():
    from courseforge.pdf_extract import select_page_range
    from test_pdf_extract import section_doc
    sel = select_page_range(section_doc(), 2, 3)
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
    client = FakeClient("{not json", "{not json", "{not json")
    with pytest.raises(LLMOutputError):
        generate_llm_course(select_page_range(section_doc(), 2, 3), "fiction.pdf", config=CONFIG, client=client)
    text = turn_text(client.requests[0], 0)
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
    retry = client.requests[1]
    assert roles(retry) == ["user", "model", "user"]
    assert turn_text(retry, 1) == "{not json"
    assert "not valid JSON" in turn_text(retry, 2)


def test_schema_errors_are_fed_back():
    bad = json.loads(course_json())
    bad["lessons"][2]["quiz"][0]["correct_option"] = "E"
    client = FakeClient(json.dumps(bad), course_json())
    result = run(client)
    assert result.verified and result.attempts == 2
    assert "lessons 3 > quiz 1: correct_option 'E'" in turn_text(client.requests[1], 2)


def test_persistently_invalid_output_stops_after_three_attempts():
    four_lessons = json.dumps({"lessons": json.loads(course_json())["lessons"][:4]})
    client = FakeClient(four_lessons, four_lessons, four_lessons, four_lessons)
    with pytest.raises(LLMOutputError, match="after 3 attempt") as info:
        run(client)
    assert len(client.requests) == 3  # never more than three generation calls
    assert info.value.problems and info.value.problems[0].startswith("lessons:")


def test_attempt_limit_cannot_be_raised_by_callers():
    bad = "{not json"
    client = FakeClient(bad, bad, bad, bad, bad)
    with pytest.raises(LLMOutputError):
        run(client, max_attempts=5)
    assert len(client.requests) == 3


def test_unexpected_top_level_fields_are_rejected():
    data = json.loads(course_json())
    data["metadata"] = {"title": "model-made metadata"}
    client = FakeClient(json.dumps(data), course_json())
    assert run(client).attempts == 2
    assert 'only a "lessons" list' in turn_text(client.requests[1], 2)


def test_empty_response_is_retried_without_an_empty_model_turn():
    client = FakeClient("", course_json())
    assert run(client).attempts == 2
    assert roles(client.requests[1]) == ["user", "user"]


# ---------------------------------------------------------------- citation failures

def test_citation_failure_gets_a_corrective_retry():
    wrong = lesson(2, 2, quote="Glowmoss spreads by releasing golden seeds into the sea.")
    client = FakeClient(course_json(lesson1=wrong), course_json())
    result = run(client)
    assert result.verified and result.attempts == 2
    assert "Lesson 2: quote not found on page 2" in turn_text(client.requests[1], 2)


def test_citation_to_a_page_outside_the_selection_is_reported():
    outside = lesson(1, 1)
    outside["sources"] = [{"page": 9, "quote": QUOTES[1]}]
    client = FakeClient(course_json(lesson0=outside), course_json())
    run(client)
    assert "page 9 does not exist" in turn_text(client.requests[1], 2)


def test_uncited_example_is_a_problem_for_ai_output():
    uncited = lesson(1, 1, example_sources=[])
    client = FakeClient(course_json(lesson0=uncited), course_json())
    run(client)
    assert "Lesson 1, example 1: has no source reference" in turn_text(client.requests[1], 2)


def test_persistent_citation_failure_returns_an_explicitly_unverified_draft():
    wrong = course_json(lesson4=lesson(5, 4, quote="Warm summers make the moss glow brighter."))
    client = FakeClient(wrong, wrong, wrong)
    result = run(client)
    assert len(client.requests) == 3
    assert not result.verified and result.attempts == 3
    assert len(result.problems) == 4 and all("quote not found on page 4" in p for p in result.problems)
    bad = [c for c in result.citations.checks if c.location.startswith("Lesson 5")]
    assert bad and not any(c.verified for c in bad)
    assert result.status_message.count("UNVERIFIED DRAFT") == 1


# ---------------------------------------------------------------- configuration

def test_missing_provider_means_ai_is_off():
    with pytest.raises(LLMConfigError, match="LLM_PROVIDER is not set"):
        load_config({})


@pytest.mark.parametrize("env,message", [
    ({"LLM_PROVIDER": "gemini"}, "GEMINI_API_KEY is not set"),
    ({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "   "}, "GEMINI_API_KEY is not set"),
    ({"LLM_PROVIDER": "anthropic", "GEMINI_API_KEY": SECRET}, "uses only the Google Gemini free tier"),
    ({"LLM_PROVIDER": "openai", "GEMINI_API_KEY": SECRET}, "not supported"),
    ({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": SECRET, "COURSEFORGE_MODEL": "gemini-4-argon"},
     "not an approved Gemini free-tier model"),
    ({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": SECRET, "COURSEFORGE_MODEL": "gemini-2.0-flash"},
     "not an approved Gemini free-tier model"),
])
def test_incomplete_or_disallowed_configuration_is_reported(env, message):
    with pytest.raises(LLMConfigError, match=message) as info:
        load_config(env)
    assert SECRET not in str(info.value)


def test_anthropic_key_alone_does_not_enable_ai():
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY is not set"):
        load_config({"LLM_PROVIDER": "gemini", "ANTHROPIC_API_KEY": "sk-ant-anything"})


def test_complete_configuration_uses_the_default_free_tier_model():
    config = load_config({"LLM_PROVIDER": " Gemini ", "GEMINI_API_KEY": SECRET})
    assert (config.provider, config.model) == ("gemini", "gemini-3.8-flash")
    assert config.api_key == SECRET and SECRET not in repr(config)


def test_explicit_approved_model_is_accepted():
    config = load_config({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": SECRET,
                          "COURSEFORGE_MODEL": "gemini-3.8-flash"})
    assert config.model == "gemini-3.8-flash"


def test_unavailable_model_is_rejected_before_any_generation():
    client = FakeClient(course_json(), model_error=api_error(404, "NOT_FOUND", "models/x is not found"))
    with pytest.raises(LLMConfigError, match="model 'gemini-3.8-flash' is not available"):
        run(client)
    assert client.requests == []


# ---------------------------------------------------------------- API errors

QUOTA_DETAILS = [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
     "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                     "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "10"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"},
]
DAILY_DETAILS = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                  "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                  "quotaValue": "250"}]}]
ZERO_DETAILS = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                 "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                 "quotaValue": "0"}]}]
KEY_DETAILS = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]


@pytest.mark.parametrize("error,expected,message", [
    (api_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", KEY_DETAILS),
     LLMAuthError, "rejected the API key"),
    (api_error(403, "PERMISSION_DENIED"), LLMAuthError, "refused access"),
    (api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", QUOTA_DETAILS), LLMRateLimitError,
     "about 37 seconds"),
    (api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", DAILY_DETAILS), LLMRateLimitError,
     "midnight Pacific time"),
    (api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", ZERO_DETAILS), LLMRateLimitError,
     "no free-tier quota"),
    (api_error(400, "INVALID_ARGUMENT", "Request contains an invalid argument."), LLMRequestError,
     "rejected the request: Request contains"),
    (api_error(500, "INTERNAL"), LLMServiceError, "HTTP 500"),
    (api_error(503, "UNAVAILABLE", "The model is overloaded."), LLMServiceError, "overloaded"),
    (httpx.ConnectTimeout("timed out"), LLMTimeoutError, "did not respond in time"),
    (httpx.ReadTimeout("timed out"), LLMTimeoutError, "did not respond in time"),
    (httpx.ConnectError("connection refused"), LLMConnectionError, "Could not connect"),
])
def test_api_errors_become_clear_messages_and_stop(error, expected, message):
    client = FakeClient(error, course_json())
    with pytest.raises(expected, match=message) as info:
        run(client)
    assert SECRET not in str(info.value)
    assert len(client.requests) == 1  # stopped: no retry after an API error


def test_quota_error_says_it_will_not_switch_providers():
    client = FakeClient(api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", QUOTA_DETAILS))
    with pytest.raises(LLMRateLimitError) as info:
        run(client)
    assert info.value.retry_after == "37"
    assert "Google sets these limits" in str(info.value)
    assert "will not retry automatically or switch to another provider" in str(info.value)


def test_key_in_a_provider_error_message_is_redacted():
    client = FakeClient(api_error(400, "INVALID_ARGUMENT", f"bad request for key {SECRET}"))
    with pytest.raises(LLMRequestError) as info:
        run(client)
    assert SECRET not in str(info.value) and "[redacted]" in str(info.value)


def test_errors_are_not_chained_to_the_raw_sdk_exception():
    client = FakeClient(api_error(400, "INVALID_ARGUMENT", "Request contains an invalid argument."))
    with pytest.raises(LLMRequestError) as info:
        run(client)
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_authentication_error_while_checking_the_model():
    client = FakeClient(model_error=api_error(400, "INVALID_ARGUMENT", "API key not valid.", KEY_DETAILS))
    with pytest.raises(LLMAuthError):
        run(client)
    assert client.requests == []


@pytest.mark.parametrize("finish,expected", [
    ("MAX_TOKENS", "cut off"),
    ("RECITATION", "recitation check"),
    ("SAFETY", "safety filter"),
    ("PROHIBITED_CONTENT", "safety filter"),
    ("OTHER", "stopped before finishing"),
])
def test_unfinished_responses_are_not_retried(finish, expected):
    client = FakeClient(gemini_response(course_json(), finish=finish), course_json())
    with pytest.raises(LLMOutputError, match=expected):
        run(client)
    assert len(client.requests) == 1


def test_blocked_prompt_is_reported():
    client = FakeClient(blocked_response("SAFETY"))
    with pytest.raises(LLMOutputError, match="no answer for these pages"):
        run(client)


# ---------------------------------------------------------------- input limits

def test_oversized_source_is_rejected_before_calling_the_api(monkeypatch):
    monkeypatch.setattr(llm, "MAX_SOURCE_CHARS", 100)
    client = FakeClient(course_json())
    with pytest.raises(LLMRequestError, match="limit is 100"):
        run(client)
    assert client.requests == [] and client.model_checks == []


def test_chapter_without_text_is_rejected_before_calling_the_api():
    client = FakeClient(course_json())
    doc, sel = chapter_selection([None, None])
    with pytest.raises(InsufficientSourceError):
        generate_llm_course(sel, doc.filename, config=CONFIG, client=client)
    assert client.requests == []


# ---------------------------------------------------------------- modes

@pytest.fixture
def no_ai_environment(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda **kwargs: False)  # ignore any local .env
    for name in ("LLM_PROVIDER", "GEMINI_API_KEY", "COURSEFORGE_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_auto_mode_without_configuration_explains_instead_of_using_demo(no_ai_environment):
    doc, sel = chapter_selection()
    with pytest.raises(LLMConfigError, match="LLM_PROVIDER is not set"):
        generate_course(sel, doc.filename, mode="auto")


def test_llm_mode_without_configuration_raises(no_ai_environment):
    doc, sel = chapter_selection()
    with pytest.raises(LLMConfigError):
        generate_course(sel, doc.filename, mode="llm")


def test_demo_mode_never_calls_the_api():
    client = FakeClient()
    doc, sel = chapter_selection()
    result = generate_course(sel, doc.filename, mode="demo", config=CONFIG, client=client)
    assert isinstance(result, DemoResult) and client.requests == [] and client.model_checks == []


def test_auto_mode_uses_gemini_when_configured():
    client = FakeClient(course_json())
    doc, sel = chapter_selection()
    result = generate_course(sel, doc.filename, mode="auto", config=CONFIG, client=client)
    assert isinstance(result, LLMResult) and result.verified


def test_failed_ai_request_is_never_replaced_by_demo_output():
    client = FakeClient(api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", QUOTA_DETAILS))
    doc, sel = chapter_selection()
    with pytest.raises(LLMRateLimitError):
        generate_course(sel, doc.filename, mode="auto", config=CONFIG, client=client)


# ---------------------------------------------------------------- real client construction

def test_real_client_is_built_with_sdk_retries_switched_off():
    client = llm.make_client(CONFIG)  # constructing the SDK client makes no request
    options = client._api_client._http_options
    assert options.retry_options is not None and options.retry_options.attempts == 1
    assert options.timeout == 300_000


def test_provider_uses_the_supplied_client():
    fake = FakeClient()
    provider = llm.make_provider(CONFIG, fake)
    assert isinstance(provider, GeminiProvider) and provider.client is fake
