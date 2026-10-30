import math
import json
import ollama import Client

#------------------------------
# 1. define math tools
# --------------------------------

def add (a:float, b: float) -> float:
    """ adds two number a + b """
    return a+b

def subtract (a:float, b: float) -> float:
    """ subtracrt two number a - b """
    return a-b

def multiply (a:float, b: float) -> float:
    """ multiply  two number a * b """
    return a*b

def divide (a:float, b: float) -> str | float | int:
    """ divide two number a / b """
    if b ==0:
        raise ValueError( "cannot divide by zero")
    return a/b

def square_root( number: float) -> float:
    pass

#----------------------------------
#2 . Create Math Agent
# ---------------------------------

class MathAgent:
    def __init__(self,model="qwen3:4b"):
        self.client = Client(host= "http://localhost:11434")
        self.model = model
        self.messages = []
        self.system_prompt = """You are a math agent , use tools to calculate and explain the answer. 
                                if any query apart form math is being asked
                                please decaline the request gracefully """
        self.tools = {
            "add": add,
            "subtract" : subtract,
            "multiply" : multiply,
            "divide" : divide
        }

    #----------------------------------
    #3 . Execute tool
    # ---------------------------------
    def execute_tool(self,tool_call):
        name = call.function.name
        arguments = call.function.arguments

        print(f"Tool : {name} ({arguments}) = {result}")

        if name not in self.tools:
            return{"error" : f "Unknown tool : {name}"}

        # function executions
        try:
            if isinstance(arguments,str):
                arguments = json.loads(arguments)

            if not isinstance(arguments,dict):
                raise ValueError("arguments must be an object")

            result = self.tools[name](**arguments)
            print(f"[Result] {result}")

            return {'result': result}
        except Exception as ex :
            return {"error" : str(ex)}


# -------------------------
# 4. Agent Reasoning Loop
#----------------------------

def run ( self , query , max_step = 5):
            # model inputs 
    self.messages.append({"role": "system", "content": self.system_prompt })
    self.messages.append( {"role": "user" , "content": query})

    for step in range(max_step):
        print( f"\n ------------- agent step {step +1} ------------------------")

        response = self.client.chat(
            model = self.model,
            messages = self.messages,
            tools = list(self.tools.values())
        )

        message = response.message
        # save assistant message including tool calls 

        self.messages.append( message.model_dump(exclude_none=True))

        # TBD 

        