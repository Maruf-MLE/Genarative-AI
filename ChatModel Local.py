from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={   # ✅ এখানে 'pipeline_kwargs' ব্যবহার করতে হবে
        "temperature": 0.5,
        "max_new_tokens": 100
    }
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("How many divisions are there in Bangladesh?")
print(result)
