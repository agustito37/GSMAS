from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class Tool(ABC):
    """One invocable capability. Subclasses define the contract the LLM sees
    (name, description, JSON-schema parameters) and implement run()."""

    name: str
    description: str
    parameters: dict  # JSON schema of the arguments

    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> str:
        """Execute and return a TEXTUAL result (it goes back into the LLM context).

        The (*args, **kwargs) signature is deliberate: each tool declares its own
        named parameters (the JSON schema is the real contract) and the registry
        dispatches dynamically via run(**arguments)."""

    def spec(self) -> dict:
        # OpenAI function-calling format. strict=True so the tool can be offered
        # ALONGSIDE structured outputs (the .parse() path rejects non-strict tools);
        # strict requires additionalProperties=false and every property listed in
        # `required` (all tools here declare their params required).
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {**self.parameters, "additionalProperties": False},
            },
        }


class ToolRegistry:
    """The system-wide tool catalog: name -> tool. Roles receive the registry and
    choose what to invoke; nothing restricts which tools a role may call."""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    def specs(self) -> list[dict]:
        return [tool.spec() for tool in self._tools.values()]

    async def run(self, name: str, arguments: dict) -> str:
        """Run a tool by name. Failures come back as TEXT, not exceptions: the model
        reads the error and adapts (retries, changes arguments, picks another tool)."""
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'"
        try:
            return await tool.run(**arguments)
        except Exception as exc:  # noqa: BLE001 - the model is the error handler here
            return f"error running '{name}': {exc}"

    def with_tool(self, tool: Tool) -> ToolRegistry:
        """A new registry with `tool` added; the base catalog is not mutated, so a
        per-judgment tool never leaks into other agents."""
        return ToolRegistry([*self._tools.values(), tool])
