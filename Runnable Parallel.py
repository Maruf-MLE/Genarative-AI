from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V3.2-Exp",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template ='Genarate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template ='Genarate a linkdin post about {topic}',
    input_variables=['topic']
)

parser= StrOutputParser()

parallel_chain= RunnableParallel({
    'tweet': RunnableSequence(prompt1,model,parser),
    'linkden':RunnableSequence(prompt2,model,parser)
})
result=parallel_chain.invoke({'topic':'AI'})

print(result)
