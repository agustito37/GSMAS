from collections.abc import Iterable

from pydantic import BaseModel

from core.providers.base import LLMProvider, LLMResponse


class MockProvider(LLMProvider):
    """Returns predefined responses in order. Lets tests run the full agent/event
    flow without spending tokens or needing an API key."""

    def __init__(self, responses: Iterable[LLMResponse | str]) -> None:
        self._responses: list[LLMResponse] = [
            r if isinstance(r, LLMResponse) else LLMResponse(content=r) for r in responses
        ]
        self._calls = 0
        self.tools_seen: list[list[dict] | None] = []  # the tools passed on each call

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        if self._calls >= len(self._responses):
            raise AssertionError("MockProvider ran out of predefined responses")
        self.tools_seen.append(tools)
        response = self._responses[self._calls]
        self._calls += 1
        return response
