from src.agents.base_agent import BaseAgent
from src.agents.prompts import MATH_AGENT_PROMPT
from src.agents.tools.math_tools import MATH_TOOLS


class MathAgent(BaseAgent):
    def __init__(self, llm, max_steps=5):
        super().__init__(
            llm=llm,
            system_prompt=MATH_AGENT_PROMPT,
            tools=MATH_TOOLS,
            max_steps=max_steps,
        )
