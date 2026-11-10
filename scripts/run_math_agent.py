"""
Run from the project root:
    uv run python -m scripts.run_math_agent
"""

from src.agents.math_agent import MathAgent
from src.config import get_settings
from src.factory import build_agent_llm


def main():
    settings = get_settings()
    agent = MathAgent(build_agent_llm(settings), max_steps=settings["agent_max_steps"])

    while True:
        question = input("\nYou: ")
        if question.strip().lower() == "exit":
            break
        print(f"Agent: {agent.run(question)}")


if __name__ == "__main__":
    main()
