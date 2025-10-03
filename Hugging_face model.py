from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "zai-org/GLM-4.6",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)
result=model.invoke("whats your name")
print(result)