import json

from core.agents.base import Agent
from core.providers.base import LLMProvider, LLMResponse
from core.tools.base import ToolRegistry

MAX_TOOL_ITERATIONS = 8  # LLM turns per agent run; overflow raises -> fail/retry bounds the spend


async def run_tool_loop(
    provider: LLMProvider,
    tools: ToolRegistry | None,
    agent: Agent,
    messages: list[dict],
    response_schema: type | None = None,
) -> LLMResponse:
    """Drive the loop: the model invokes tools until it emits its final answer.
    agent.messages is the STM (the whole exchange lives there and dies with the
    agent). Tool failures return to the model as text (it adapts); a runaway loop
    raises at MAX_TOOL_ITERATIONS, the worker fails the agent, and the retry budget
    bounds the total spend."""
    agent.messages = list(messages)
    specs = tools.specs() if tools else None
    for _ in range(MAX_TOOL_ITERATIONS):
        response = await provider.complete(
            agent.messages, tools=specs, response_schema=response_schema
        )
        if not response.tool_calls:
            return response
        agent.messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in response.tool_calls
                ],
            }
        )
        for call in response.tool_calls:
            result = (
                await tools.run(call.name, call.arguments)
                if tools
                else f"error: no tools available (requested '{call.name}')"
            )
            agent.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    raise RuntimeError(f"tool loop exceeded MAX_TOOL_ITERATIONS ({MAX_TOOL_ITERATIONS})")
