from src.rag.core.generation.base_generator import BaseGenerator
from src.rag.prompts import GROUNDED_ANSWER_PROMPT, NO_ANSWER


class GroundedGenerator(BaseGenerator):
    """Answers only from the given chunks and cites their ids."""

    def __init__(self, llm, instructions=GROUNDED_ANSWER_PROMPT):
        super().__init__(llm, instructions)

    def generate(self, query, chunks):
        if not chunks:
            return NO_ANSWER

        user_input = self._build_user_input(query, chunks)
        answer = self.llm.ask(self.instructions, user_input)

        return answer or NO_ANSWER
