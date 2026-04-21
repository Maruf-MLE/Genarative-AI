from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse 
from Self_rag_py import rag_app


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = False,
    allow_methods = ["*"],
    allow_headers = ['*']
)

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

class UserQuery(BaseModel):
    question: Annotated[str,Field(...,min_length=1,description='please give the question here')] 

@app.post('/ask')
async def ask_query(query:UserQuery):
    try:
        question = query.question

        result = rag_app.invoke({
            'question':question,
            "need_retrieval": False,
            'relevant_docs':[],
            "docs": [],
            'retries':0,
            'issup': '',
            'evidence':[],
            'isuse': '',
            'use_reason':'',
            'retrieval_query': '',
            'rewrite_tries':0
            
        })

        return {'answer': result['answer']}

    except Exception as e:
        return {'error': str(e)}