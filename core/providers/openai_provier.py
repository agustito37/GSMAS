import json
from openai import AsyncOpenAI
from pydantic import BaseModel
from core.providers.base import LLMProvider, LLMResponse, ToolCall

class OpenAIProvider(LLMProvider):
    """LLMProvider based on the official OpenAI SDK (async)."""

    def __init__(self, model: str, api_key: str | None = None, max_retries: int = 3) -> None:
        # api_key=None -> the SDK reads OPENAI_API_KEY from the environment.
        # max_retries uses the SDK's built-in exponential backoff — no need to
        # hand-roll retries.
        self._client = AsyncOpenAI(api_key=api_key, max_retries=max_retries)
        self._model = model

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> LLMResponse:
        opts = {"model": self._model, "messages": messages}
        if tools:
            opts["tools"] = tools

        if response_schema is not None:
            # structured output: the SDK validates straight into the Pydantic model
            completion = await self._client.beta.chat.completions.parse(
                response_format=response_schema, **opts
            )
        else:
            completion = await self._client.chat.completions.create(**opts)

        message = completion.choices[0].message
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (message.tool_calls or [])
        ]
        usage = {
            "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
            "output_tokens": completion.usage.completion_tokens if completion.usage else 0,
        }
        return LLMResponse(content=message.content or "", tool_calls=tool_calls, usage=usage)
