import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  

llm = ChatOpenAI(
    model="z-ai/glm-4.5-air:free", 
    api_key=os.getenv("OPENROUTER_API_KEY"),  
    base_url="https://openrouter.ai/api/v1"
)

response = llm.invoke("give me e poem for 40 line")
print(response)
