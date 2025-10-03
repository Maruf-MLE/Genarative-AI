from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage
#Chat template
Chat_template=ChatPromptTemplate([
    ('system',"You are a helpful customer support agent"),
    (MessagesPlaceholder(variable_name='chat_history')),
    ('human','{query}')
])
chat_history=[]


# load chat history
with open('chat_history.txt') as f:
    chat_history.append(f.readlines())

print(chat_history)

# creat Prompt
prompt=Chat_template.invoke({'chat_history':chat_history,'query':'Where is my refund'})

print(prompt)