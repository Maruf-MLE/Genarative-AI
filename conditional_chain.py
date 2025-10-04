from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V3.2-Exp",
    task = "text-generation"
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

class feedback(BaseModel):
    sentiment:Literal['positive','negative']= Field(description='Give the sentiment of the feedback')

parser2= PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template='classify the sentiment of the folowing d=feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)


classifier_chain= prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an a appropriate response to this postive feedback \n {feedback}',
    input_variables=['feedback'],
   
)

prompt3 = PromptTemplate(
    template='Write an a appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback'],
   
)

branch_chain= RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2 | model | parser),
    (lambda x:x.sentiment=='negative',prompt2 | model | parser),
    RunnableLambda(lambda x: 'coud not find sentiment')
)

chain = classifier_chain | branch_chain

result=chain.invoke({'feedback':'this is a very good phone'})

print(result)
