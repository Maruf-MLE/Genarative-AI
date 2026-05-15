from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Rag/Feature Engineering for Machine Learning.pdf')

docs=loader.load()

spliter = CharacterTextSplitter(
    chunk_size =100,
    chunk_overlap = 0,
    separator= ''
)

result=spliter.split_documents(docs)

print(result[0])