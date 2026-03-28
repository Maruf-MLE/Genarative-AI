from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence



load_dotenv()




loader = PyPDFLoader('Rag/Feature Engineering for Machine Learning.pdf')
docs= loader.load()

print(len(docs))