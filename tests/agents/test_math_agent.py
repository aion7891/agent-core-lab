from src.agents.math_agent import MathAgent
from src.agents.tools.math_tools import divide
from tests.fakes import FakeLLM


def tool_call(name, **arguments):
    return {"content": "", "tool_calls": [{"name": name, "arguments": arguments}]}


def test_agent_calls_tools_then_answers():
    llm = FakeLLM([
        tool_call("multiply", a="20", b="5"),
        tool_call("add", a=100, b=10),
        "The answer is 110.",
    ])
    agent = MathAgent(llm)

    assert agent.run("20 times 5, then add 10") == "The answer is 110."

    tool_messages = [m for m in agent.messages if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages] == ["100.0", "110.0"]


def test_unknown_tool_and_bad_args_do_not_crash():
    llm = FakeLLM([tool_call("power", a=2), tool_call("add", a="x", b=1), "done"])
    agent = MathAgent(llm)

    assert agent.run("?") == "done"
    tool_messages = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert tool_messages[0] == "Error: unknown tool 'power'"
    assert tool_messages[1].startswith("Error while running add")


def test_max_steps_stops_the_loop():
    llm = FakeLLM([tool_call("add", a=1, b=1)] * 2)
    assert MathAgent(llm, max_steps=2).run("loop") == "Maximum steps reached without a final answer."


def test_divide_by_zero():
    assert divide(1, 0).startswith("undefined")


def test_system_prompt_added_once_across_runs():
    agent = MathAgent(FakeLLM(["a", "b"]))
    agent.run("first")
    agent.run("second")
    assert [m["role"] for m in agent.messages].count("system") == 1
