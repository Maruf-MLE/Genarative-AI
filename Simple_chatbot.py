from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage


load_dotenv()

chat_history = []

llm = HuggingFaceEndpoint(
    repo_id= "zai-org/GLM-4.6",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

message= [
    SystemMessage(content='You are a teacher'),
    ]


while True:
    user_input = input('you: ')

    message.append(HumanMessage(content=user_input))
  
    if user_input == 'exit':
        break
    result = model.invoke(message)
    message.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(message)