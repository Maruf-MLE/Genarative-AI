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
prompt1 = PromptTemplate(
    template= 'write a joke about {topic}',
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template='Explain the foloowing joke - {text}',
    input_variables=['text']
)
parser = StrOutputParser()
chain= RunnableSequence(prompt1,model,parser,prompt2,model,parser)
print(chain.invoke({'topic':'AI'}))