"""A LangChain chat model that runs prompt-based tool calling over any LLMClient."""
import json
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

PROTOCOL = """You are running inside an automated agent loop. Reply ONLY with a JSON object matching the response schema.
- To call a tool: {"action": "call_tool", "tool": "<tool name>", "args": {<arguments>}}. Call exactly one tool per reply.
- To finish: {"action": "final", "final": <your final answer object>}.
Base every statement on tool results you have actually received."""


class LLMOutputError(RuntimeError):
    """The model did not produce a valid action."""


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content)


class CLIChatModel(BaseChatModel):
    client: Any
    tool_specs: list[dict] = Field(default_factory=list)
    final_schema: dict = Field(default_factory=lambda: {"type": "object"})

    @property
    def _llm_type(self) -> str:
        return "cli-chat-model"

    def bind_tools(self, tools, *, final_schema: dict | None = None, **kwargs: Any) -> "CLIChatModel":
        update: dict[str, Any] = {"tool_specs": [convert_to_openai_tool(t)["function"] for t in tools]}
        if final_schema is not None:
            update["final_schema"] = final_schema
        return self.model_copy(update=update)

    def response_schema(self) -> dict:
        actions = ["final"] + (["call_tool"] if self.tool_specs else [])
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": actions},
                "tool": {"type": "string"},
                "args": {"type": "object"},
                "final": self.final_schema,
            },
            "required": ["action"],
        }

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        system, prompt = self._render(messages)
        schema = self.response_schema()
        error: str | None = None
        for _ in range(2):
            attempt_prompt = prompt if error is None else (
                f"{prompt}\n\nYOUR PREVIOUS REPLY WAS INVALID: {error}\nReply again with a valid JSON object."
            )
            result = self.client.complete(system, attempt_prompt, json_schema=schema)
            try:
                message = self._to_message(result.structured)
            except LLMOutputError as exc:
                error = str(exc)
                continue
            return ChatResult(generations=[ChatGeneration(message=message)])
        raise LLMOutputError(f"model produced an invalid reply twice: {error}")

    def _render(self, messages: list[BaseMessage]) -> tuple[str, str]:
        system_parts: list[str] = []
        lines: list[str] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                system_parts.append(_text(m.content))
            elif isinstance(m, HumanMessage):
                lines.append(f"[USER]\n{_text(m.content)}")
            elif isinstance(m, AIMessage):
                for call in m.tool_calls:
                    lines.append(f"[ASSISTANT called tool {call['name']} with args {json.dumps(call['args'])}]")
                if m.content:
                    lines.append(f"[ASSISTANT]\n{_text(m.content)}")
            elif isinstance(m, ToolMessage):
                lines.append(f"[TOOL RESULT {m.name or ''}]\n{_text(m.content)}")
        if self.tool_specs:
            tools_block = "AVAILABLE TOOLS:\n" + json.dumps(self.tool_specs, indent=2)
        else:
            tools_block = "AVAILABLE TOOLS: (no tools available — you must reply with action=final now)"
        system = "\n\n".join(system_parts + [PROTOCOL, tools_block])
        return system, "\n\n".join(lines)

    def _to_message(self, data: Any) -> AIMessage:
        if not isinstance(data, dict):
            raise LLMOutputError("reply is not a JSON object")
        action = data.get("action")
        if action == "call_tool" and self.tool_specs:
            names = {spec["name"] for spec in self.tool_specs}
            tool_name = data.get("tool")
            args = data.get("args") or {}
            if tool_name not in names:
                raise LLMOutputError(f"unknown tool {tool_name!r}; available tools: {sorted(names)}")
            if not isinstance(args, dict):
                raise LLMOutputError("args must be a JSON object")
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            return AIMessage(content="", tool_calls=[
                {"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}
            ])
        if action == "final":
            if "final" not in data:
                raise LLMOutputError("action=final requires a 'final' field")
            return AIMessage(content=json.dumps(data["final"]))
        raise LLMOutputError(f"unsupported action {action!r}")
