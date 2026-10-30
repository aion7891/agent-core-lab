from intelligence import chat_agent
from intelligence.chat_agent import ChatAgent

class ChatService:
    def __init__(self):
        self.chat_agent = ChatAgent()

    def llm_chat(self,message):
        output = self.chat_agent.chat(message)
        return output

