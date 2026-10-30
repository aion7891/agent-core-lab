from src.agent.math_agent import MathAgent

agent = MathAgent()
while True:
    question = input("\n You")
    if question.lower() == "exit":
        break
    agent_response = agent.run(question)
    print( f" Agent : {agent_response} ")
