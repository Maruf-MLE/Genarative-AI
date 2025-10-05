from langchain_community.document_loaders import DirectoryLoader,PyPDFDirectoryLoader,PyPDFLoader

loader= DirectoryLoader(
    path= 'Rag/Books',
    glob= '*.pdf',
    loader_cls=PyPDFLoader

)

# doc= loader.load()
doc = loader.lazy_load()

for i in doc:
    print(i)