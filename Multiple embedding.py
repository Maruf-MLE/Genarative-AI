from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding= OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)
document = [
    'hi',
    'how are you',
    'Whats your name'

]
result= embedding.embed_documents(document)

print(result)