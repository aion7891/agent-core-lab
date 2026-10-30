import os
from openai import OpenAI, completions
from dotenv import load_dotenv
load_dotenv()


class LLMFactory:
    def __init__(self,model_provider="openrouter", llm_model='google/gemma-4-26b-a4b-it:free'):
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"))
        self.model_provider = model_provider
        self.llm_model = llm_model

    def chat_json(self,message):
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": message
                    }
                ]
                )
            return completion
        except Exception as e:
            print(f"Error at chat json : {e}")
            return {}

from agno.agent import Agent
from agno.models.openrouter import OpenRouter



if __name__ == '__main__':
    model_provider = os.getenv("LLM_PROVIDER")
    model_name = os.getenv("LLM_MODEL")
    llmfactory = LLMFactory()
    print(llmfactory.chat_json("Hello World"))

    agent = Agent(
        model=OpenRouter(id=model_name),
        markdown=True
    )
    # Print the response in the terminal
    agent.print_response("Share a 2 sentence horror story.")


