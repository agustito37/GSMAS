from abc import ABC, abstractmethod

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict  # parsed from the model's JSON arguments


class LLMResponse(BaseModel):
    content: str
    tool_calls: list[ToolCall] = []
    usage: dict = {}  # e.g. {"input_tokens": ..., "output_tokens": ...}


class LLMProvider(ABC):
    """Provider-agnostic async LLM interface.

    The model backing an agent is a parameter of the agent, not of the
    architecture (provider-agnostic): two agents of the same role can run on
    different providers and still share the role's LTM.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse: ...
