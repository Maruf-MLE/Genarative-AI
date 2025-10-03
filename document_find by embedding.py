from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Cricket is one of the most popular sports in the world.",
    "Sachin Tendulkar is known as the God of Cricket.",
    "The ICC World Cup is the biggest tournament in cricket.",
    "Test cricket is the oldest and most traditional format of the game.",
    "T20 cricket has made the sport more exciting and fast-paced."
]

query='tell me about Sachin Tendulkar'

doc_embedding= embedding.embed_documents(documents)
query_embedding= embedding.embed_query(query)

result=cosine_similarity([query_embedding],doc_embedding) [0]
index,score= sorted(list(enumerate(result)),key=lambda x:x[1]) [-1]

print(query)
print(documents[index])
print('similarity score is :',score)
