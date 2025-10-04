from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

# HuggingFace Endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

class person(BaseModel):
    name:str=Field(description='Name of the person')
    age:int = Field(gt=18,description= 'Age of the person')
    city:str = Field(description='Name of the city person belongs to')

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template='Genarate the name ,age and city of a fictional {place} person \n {format_instruction} ',
    input_variables='place',
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
prompt= template.invoke({'place':'Bangladeshi'})

result=model.invoke(prompt)
final_result=parser.parse(result.content)

print(final_result)
