from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


# Create document

doc1 = Document(
    page_content = 'I am e cricket player',
    metadata = {'team':'Rajshahi'}
)
doc2 = Document(
    page_content = 'I am e football player',
    metadata = {'team':'Rajshahi'}
)
doc3 = Document(
    page_content = 'I am  Free fire player',
    metadata = {'team':'Rajshahi'}
)

docs= [doc1,doc2,doc3]

embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store = Chroma(
  embedding_function = embedding_function,
  persist_directory = 'chrona_db',
  collection_name = 'sample'
)

# add documents
vector_store.add_documents(docs)

# view documents
vector_store.get(include=['embeddings','documents', 'metadatas'])

# search documents
vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)

# search with similarity score
vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)

# meta-data filtering
vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)

# view  all documents
vector_store.get(include=['embeddings','documents', 'metadatas'])

# delete document
vector_store.delete(ids=['09a39dc6-3ba6-4ea7-927e-fdda591da5e4'])