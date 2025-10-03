from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
 
chatmodel=ChatOpenAI(model='omni-moderation-latest',temperature=0.3,max_completion_tokens=10)

result=chatmodel.invoke('whats your name')
print(result.content)