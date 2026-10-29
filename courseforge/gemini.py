"""Google Gemini API (free tier) adapter for CourseForge AI.

Uses the maintained google-genai SDK and the stateless generateContent endpoint, so no
conversation is stored server-side. The SDK's own retries are switched off: every call
is made exactly once, and quota or rate-limit errors stop generation immediately.

Free tier: Google sets the quotas, and content sent on the free tier may be used by
Google to improve its products. The tier is decided by the Google Cloud project behind
the API key: a project with a linked billing account is on a paid tier. CourseForge
cannot detect billing, so the key must come from a project without billing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from courseforge.ai_errors import (
    LLMAuthError, LLMConfigError, LLMConnectionError, LLMError, LLMOutputError, LLMRateLimitError,
    LLMRequestError, LLMServiceError, LLMTimeoutError,
)

try:  # the SDK can also use httpx2 when it is installed
    import httpx2
    _TIMEOUT_ERRORS: tuple = (httpx.TimeoutException, httpx2.TimeoutException)
    _TRANSPORT_ERRORS: tuple = (httpx.TransportError, httpx2.TransportError)
except ImportError:  # pragma: no cover
    _TIMEOUT_ERRORS = (httpx.TimeoutException,)
    _TRANSPORT_ERRORS = (httpx.TransportError,)

REQUEST_TIMEOUT_MS = 300_000

# JSON Schema keywords that Gemini's response_json_schema accepts (from the SDK docs).
_SUPPORTED_KEYWORDS = {
    "$id", "$defs", "$ref", "$anchor", "type", "format", "title", "description", "enum", "items",
    "prefixItems", "minItems", "maxItems", "minimum", "maximum", "anyOf", "oneOf", "properties",
    "additionalProperties", "required", "propertyOrdering",
}
_SAFETY_STOPS = {types.FinishReason.SAFETY, types.FinishReason.BLOCKLIST,
                 types.FinishReason.PROHIBITED_CONTENT, types.FinishReason.SPII}
_NORMAL_STOPS = {None, types.FinishReason.STOP, types.FinishReason.FINISH_REASON_UNSPECIFIED}


@dataclass
class ProviderReply:
    text: str
    usage: dict[str, int] = field(default_factory=dict)


def make_gemini_client(api_key: str) -> genai.Client:
    """A client that never retries on its own (attempts=1 disables the SDK's retry loop)."""
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        timeout=REQUEST_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)))


def gemini_schema(model: type[BaseModel]) -> dict:
    """The model's JSON schema reduced to the keywords Gemini supports.

    Dropped rules (string lengths, patterns, defaults) are still enforced afterwards by
    parse_course(). Every property is marked required so the model always returns the
    source lists that citation checking needs.
    """
    def clean(node):
        if isinstance(node, list):
            return [clean(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:  # Gemini allows nothing but $-keywords next to $ref
            return {k: v for k, v in node.items() if k.startswith("$")}
        out = {}
        for key, value in node.items():
            if key not in _SUPPORTED_KEYWORDS:
                continue
            if key in ("properties", "$defs"):
                out[key] = {name: clean(sub) for name, sub in value.items()}
            else:
                out[key] = clean(value)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return clean(model.model_json_schema())


class GeminiProvider:
    """Calls one Gemini model. Raises only courseforge.ai_errors exceptions."""

    name = "gemini"

    def __init__(self, model: str, api_key: str, client=None):
        self.model = model
        self._api_key = api_key
        self.client = client if client is not None else make_gemini_client(api_key)

    def verify_model(self) -> None:
        """Check that the model exists for this key (a metadata request; no tokens are generated)."""
        try:
            self.client.models.get(model=self.model)
        except Exception as exc:
            raise self._translate(exc) from None

    def generate(self, system: str, turns: list[dict], output_model: type[BaseModel],
                 max_output_tokens: int) -> ProviderReply:
        """One generateContent call. turns are {"role": "user" | "model", "text": str}."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=gemini_schema(output_model),
            max_output_tokens=max_output_tokens,
            candidate_count=1,
        )
        contents = [types.Content(role=t["role"], parts=[types.Part(text=t["text"])]) for t in turns]
        try:
            response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        except Exception as exc:
            raise self._translate(exc) from None

        meta = response.usage_metadata
        usage = {"input_tokens": (meta.prompt_token_count or 0) if meta else 0,
                 "output_tokens": (meta.candidates_token_count or 0) if meta else 0,
                 "thinking_tokens": (meta.thoughts_token_count or 0) if meta else 0}
        if not response.candidates:
            feedback = response.prompt_feedback
            reason = feedback.block_reason.name if feedback and feedback.block_reason else "no response"
            raise LLMOutputError(f"Gemini returned no answer for these pages ({reason}). Try demo mode.")
        candidate = response.candidates[0]
        reason = candidate.finish_reason
        if reason == types.FinishReason.MAX_TOKENS:
            raise LLMOutputError("The Gemini response was too long and was cut off. Select fewer pages.")
        if reason == types.FinishReason.RECITATION:
            raise LLMOutputError("Gemini stopped because its answer repeated the source text too "
                                 "closely (Google's recitation check). Try again, or select fewer pages.")
        if reason in _SAFETY_STOPS:
            raise LLMOutputError("Gemini declined to process these pages (safety filter). Try demo mode.")
        if reason not in _NORMAL_STOPS:
            raise LLMOutputError(f"Gemini stopped before finishing ({reason.name}). Try again later.")
        parts = candidate.content.parts if candidate.content and candidate.content.parts else []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        return ProviderReply(text=text, usage=usage)

    # ------------------------------------------------------------ errors

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[redacted]") if self._api_key else text

    def _translate(self, exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, _TIMEOUT_ERRORS):
            return LLMTimeoutError("Gemini did not respond in time. Try again, or select fewer pages.")
        if isinstance(exc, _TRANSPORT_ERRORS):
            return LLMConnectionError("Could not connect to the Gemini API. Check the internet connection.")
        if not isinstance(exc, genai_errors.APIError):
            return LLMServiceError("The Gemini request failed unexpectedly.")

        code = exc.code or 0
        message = self._redact(exc.message or "")
        details = _error_details(exc)
        reasons = {d.get("reason") for d in details if isinstance(d, dict)}
        if code == 429 or exc.status == "RESOURCE_EXHAUSTED":
            return _quota_error(details)
        if "API_KEY_INVALID" in reasons or code == 401 or (code == 400 and "api key" in message.lower()):
            return LLMAuthError("Gemini rejected the API key. Check GEMINI_API_KEY. Keys created "
                                "before September 2026 may be Standard keys, which the Gemini API "
                                "no longer accepts; create a new key in Google AI Studio.")
        if code == 403:
            return LLMAuthError("Gemini refused access (permission denied). The key may be restricted, "
                                "revoked or an outdated Standard key; create a new key in Google AI Studio.")
        if code == 404:
            return LLMConfigError(f"The model '{self.model}' is not available to this API key.")
        if code == 400:
            return LLMRequestError(f"Gemini rejected the request: {message or 'invalid request'}")
        if code >= 500:
            return LLMServiceError(f"Gemini is unavailable or overloaded (HTTP {code}). Try again later.")
        return LLMServiceError(f"Gemini returned an error (HTTP {code}).")


def _error_details(exc: genai_errors.APIError) -> list:
    body = exc.details if isinstance(exc.details, dict) else {}
    inner = body.get("error", body)
    details = inner.get("details", []) if isinstance(inner, dict) else []
    return details if isinstance(details, list) else []


def _quota_error(details: list) -> LLMRateLimitError:
    retry_after = None
    daily = zero_limit = False
    for detail in details:
        if not isinstance(detail, dict):
            continue
        delay = detail.get("retryDelay")
        if isinstance(delay, str) and (m := re.fullmatch(r"(\d+(?:\.\d+)?)s", delay)):
            retry_after = str(round(float(m.group(1))))
        for violation in detail.get("violations", []) or []:
            quota_id = str(violation.get("quotaId", ""))
            daily = daily or "PerDay" in quota_id
            zero_limit = zero_limit or str(violation.get("quotaValue", "")) == "0"
    if zero_limit:
        advice = " This model has no free-tier quota for your project."
    elif daily:
        advice = " The daily free-tier limit was reached; it resets at midnight Pacific time."
    elif retry_after:
        advice = f" Try again in about {retry_after} seconds."
    else:
        advice = " Try again later."
    return LLMRateLimitError(
        "The Gemini free-tier quota or rate limit was reached (Google sets these limits)." + advice
        + " CourseForge stopped and will not retry automatically or switch to another provider.",
        retry_after)
