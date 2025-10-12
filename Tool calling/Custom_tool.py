from langchain_core.tools import tool

# Create e funtion
@tool
def multiply(a:int,b:int) -> int:
    """Mutliply tow numbers"""
    return a*b

result = multiply.invoke({'a':5,'b':6})

print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)
