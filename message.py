from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "zai-org/GLM-4.6",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

message= [
    SystemMessage(content='you name Maruf'),
    HumanMessage(content='tell me about langchain')
    
]
result = model.invoke(message)
message.append(AIMessage(content=result.content))
print(message)