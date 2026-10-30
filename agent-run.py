from src.agent.math_agent import MathAgent


model_id = "gemma3:270m"
agent = MathAgent()

# memory = """ 

# You : what is 20 multiply by 5 , and then add 10
# agent is thinking .... 
#  Agent : The user's question asks for the result of the expression: **20 multiply by 5, and then add 10**.

# To solve this using the provided tools:

# 1. **Multiply 20 and 5**:
#    - This gives $ 20 \times 5 = 100 $.
#    - The tool call for this step is:  
#      ```json
#      {"name": "multiply", "arguments": {"a": 20, "b": 5}}
#      ```

# 2. **Add the result (100) and 10**:
#    - This gives $ 100 + 10 = 110 $.
#    - The tool call for this step is:  
#      ```json
#      {"name": "add", "arguments": {"a": 100, "b": 10}}
#      ```

# After executing both tool calls, the final result is **110**.

# Thus, the final answer is:

# $$
# \boxed{110}
# $$ 

# """

while True:
    question = input("\n You : ")
    if question.lower() == "exit":
        break
    # input =f" previous memory : {memory}  \n \n QUERY : {question} "
    agent_response = agent.run(question)
    print( f" Agent : {agent_response} ")
