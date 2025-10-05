from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V3.2-Exp",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)


loader = TextLoader('Rag/Cricket.txt')
docs=loader.load()

prompt= PromptTemplate(
    template='Write a summary for the following poem -\n {poem}',
    input_variables=['poem']
)

parser=StrOutputParser()

# print(docs)

chain = prompt | model | parser
result= chain.invoke({'poem':docs[0].page_content})
print(result)