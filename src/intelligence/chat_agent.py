from utils.llm_factory import LLMFactory

class ChatAgent:
    def __init__(self,model):
        self.model = model

    def chat(self,message):
        message = message.lower()
        result = LLMFactory.chat_json(message)
        return result

