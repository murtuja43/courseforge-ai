"""Errors raised by AI course generation. Every message is safe to show to a teacher
and never contains the API key."""

from __future__ import annotations


class LLMError(Exception):
    """AI generation failed."""


class LLMConfigError(LLMError):
    """AI generation is not configured, or the configured model is not available."""


class LLMAuthError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    """A quota or rate limit was reached. CourseForge stops; it never retries or switches provider."""

    def __init__(self, message: str, retry_after: str | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


class LLMRequestError(LLMError):
    pass


class LLMServiceError(LLMError):
    pass


class LLMOutputError(LLMError):
    """The model did not produce an acceptable course."""

    def __init__(self, message: str, problems: list[str] | None = None):
        super().__init__(message)
        self.problems = problems or []
