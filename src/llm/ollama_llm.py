import ollama

from src.llm.base_llm import BaseLLM
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaLLM(BaseLLM):
    def __init__(self, model, host=None, think=False):
        super().__init__(model)
        self.think = think
        self.client = ollama.Client(host=host)

    def chat(self, messages, tools=None):
        logger.debug("Calling Ollama model '%s'", self.model)

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                # only send "think" when it is on, some models do not support it
                think=True if self.think else None,
            )
        except Exception as error:
            raise RuntimeError(
                f"Ollama chat failed for model '{self.model}': {error}"
            ) from error

        return {
            "content": response.message.content or "",
            "tool_calls": self._read_tool_calls(response.message),
            "message": response.message,
        }

    def tool_result_message(self, tool_call, result):
        return {
            "role": "tool",
            "tool_name": tool_call["name"],
            "content": str(result),
        }

    def _read_tool_calls(self, message):
        tool_calls = []

        for call in message.tool_calls or []:
            tool_calls.append({
                "name": call.function.name,
                "arguments": dict(call.function.arguments or {}),
            })

        return tool_calls
