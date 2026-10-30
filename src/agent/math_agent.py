# prerequesits 
# pip install ollama
# ollama pull qwen3:4b

from ollama import chat

class MathAgent:
    def __init__(self,model="qwen3:4b"):
        self.model = model
        self.messages = []
        self.system_prompt = """You are a math agent , use tools to calculate and explain the answer. 
                                if any query apart form math is being asked
                                please decaline the request gracefully """

    # Tools 
    def add (self,a:float, b: float) -> float:
        """ adds two number a + b """
        return a+b

    def subtract (self,a:float, b: float) -> float:
        """ subtracrt two number a - b """
        return a-b

    def multiply (self,a:float, b: float) -> float:
        """ multiply  two number a * b """
        return a*b

    def divide (self,a:float, b: float) -> str | float | int:
        """ divide two number a / b """
        if b ==0:
            return "undefined"
        return a/b

    # agent loops
    def run(self, query):

        # model inputs 
        self.messages.append({"role": "system", "content": self.system_prompt })
        self.messages.append( {"role": "user" , "content": query})

        tools = [self.add, self.subtract, self.multiply, self.divide]
        print( '================input for LLM ========================')
        print(self.messages)
        print('========================================\n')

        for _ in range(5):
            # LLM brain
            print(f"agent is thinking .... ")
            response = chat(
                model = self.model,
                messages = self.messages,
                tools = tools,
                think = True)
            
            self.messages.append(response.message)

            # observe
            if not response.message.tool_calls:
                return response.message.content

            for call in response.message.tool_calls:
                name = call.function.name
                args = call.function.arguments

                # function executions
                try:
                    result = getattr(self,name)(**args)
                except Exception as ex:
                    result = f"Error in execution : {ex}"
                
                print(f"Tool : {name} ({args}) = {result}")

                self.messages.append({
                    "role": "tool",
                    "tool_name" : name,
                    "content": str(result)
                })

        return "maximum steps reached"



            


            

        

