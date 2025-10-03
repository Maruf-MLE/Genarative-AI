from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# HuggingFace Endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# JSON Output Parser
parser = JsonOutputParser()

# Prompt Template
template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.format()
result = model.invoke(prompt)

final_result=parser.parse(result.content)
print(final_result)
