from langchain.tools import StructuredTool
from pydantic import BaseModel,Field

def multiply_fun(a,b):
    return a * b 

class MultiplyInput(BaseModel):
    a: int = Field(required = True,description='The first Number')
    b:int = Field(required = True,description='The 2nd number')

multiply_tool = StructuredTool.from_function(
    func = multiply_fun,
    name= 'multiply',
    description='Multiply tow numbers',
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a':5,'b':3})
print(result)

