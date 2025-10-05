from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableLambda,RunnablePassthrough,RunnableParallel,RunnableBranch


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V3.2-Exp",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt1= PromptTemplate(
    template='Write a detailed report on topic \n {topic}',
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template='summarize the following text \n {text}',
    input_variables= ['text']
)

parser= StrOutputParser()

report_gen_chain= RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>100,RunnableSequence(prompt2,model,parser)),
    (RunnablePassthrough)
)

final_chain= RunnableSequence(report_gen_chain,branch_chain)

result=final_chain.invoke({'topic':'AI'})
print(result)
