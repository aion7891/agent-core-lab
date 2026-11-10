"""
Base class for every LLM provider (Ollama, OpenAI, Anthropic, ...).

To add a new provider, create a new file in this folder, inherit from
BaseLLM, and implement chat() and tool_result_message().
"""


class BaseLLM:
    def __init__(self, model):
        self.model = model

    def chat(self, messages, tools=None):
        """
        Send a list of messages to the model.

        tools: optional list of plain Python functions the model may call.

        Must return a dict:
            {
                "content": "text reply",
                "tool_calls": [{"name": "add", "arguments": {"a": 1, "b": 2}}],
                "message": the assistant message to append to the history,
            }
        """
        raise NotImplementedError

    def tool_result_message(self, tool_call, result):
        """Build the message that sends a tool's result back to the model."""
        raise NotImplementedError

    def ask(self, instructions, user_input):
        """Shortcut for a single question with a system instruction."""
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_input},
        ]
        return self.chat(messages)["content"].strip()
