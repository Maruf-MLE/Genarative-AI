from typing import List, TypedDict, Annotated
import re
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# FIX 1: HuggingFaceEndpoint & ChatHuggingFace import added

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from langchain_core.output_parsers import StrOutputParser
splitter = RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=100)
docs = PyPDFLoader("./documents/book1.pdf").load()
split_docs = splitter.split_documents(docs)
embed_model = HuggingFaceEmbeddings(model= "sentence-transformers/all-MiniLM-L6-v2")
model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

vector_store = FAISS.from_documents(split_docs,embed_model)
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={'k':4})
# retriever sentence striper funtion
def chunks_splitter(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()

    # Step 1: sentence boundary দিয়ে আগে ভাগ করো
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)

    # Step 2: ছোট sentence গুলোকে জোড়া লাগিয়ে 200 char-এর chunk বানাও
    chunks = []
    current_chunk = ""

    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        # যদি current_chunk-এ s যোগ করলে 200 পার হয়
        if len(current_chunk) + len(s) + 1 > 100 and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = s  # নতুন chunk শুরু হবে এই sentence দিয়ে
        else:
            current_chunk = (current_chunk + " " + s).strip()

    # শেষে যা বাকি থাকবে সেটাও রাখো
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

upper_th = 0.7
lower_th = 0.3
class State(TypedDict):
    question:str
    docs:list[Document]
    ans:str

    all_strips:list[str]
    kept_strips:[list[str]]
    refined_context:str
def retrieve_node(state):
    q = state['question']
    return {'docs':retriever.invoke(q)}
class DocEvalScore(BaseModel):
    score: float
    reason: str

doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        ('system',"You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only."),
        ('human','Question :{question}\n\nchunk:\n{chunk}')
    ]
)

doc_eval_chain = doc_eval_prompt | model.with_structured_output(DocEvalScore)

def doc_eval_node(state:State) -> State:
    q = state['question']

    scores:list[float] = []
    reasons:list[str] = []
    good:list[Document] = []

    for d in state[docs]:
        out = doc_eval_chain.invoke({'question':q,'chunk':d.page_content})
        scores.append(out.score)
        reasons.append(out.reason)

        if out.score > lower_th:
            good.append(d)

    if any(s > upper_th for s in scores):
        return{
            'good_docs': good,
            'verdict': 'CORRECT',
            'reason':f'At least one retrieved chunk scored > {UPPER_TH}.'
        }
    
    if len(scores) > 0 and all(s < lower_th for s in scores):
        why = 'No chunk was sufficient'
        return {
            'good_docs': [],
            'verdict':"INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}. {why}"
        }

    why = 'Mixed relevance signals'
    return {
        'good_docs': good,
        'verdict': "AMBIGUOUS",
        'reason': f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}. {why}",
    }
    
    
filter_prompt = ChatPromptTemplate(
    [
          ("system", 
         "You are a strict relevance filter. \n"
         "Does the sentence directly help answer the question? \n"
         "Reply with ONLY one word: YES or NO. Nothing else."
        ),
        ('human','question:{question}\n\nsentence:{sentence}')
    ]
)

filter_chain = filter_prompt | model | StrOutputParser() | (lambda x:"YES" in x.strip().upper())
def refine_node(state:State) -> State:
    q = state['question']
    good_docs = state['good_docs']
    all_split_chunks = []
    all_strips = []
    all_output = []
    kept_strips = []
    all_input_for_filter = []


    for doc in good_docs:
        strips = chunks_splitter(doc.page_content)
        all_split_chunks.extend(strips)

        for s in strips:
            all_input_for_filter.append({'question':q,'sentence':s})
            all_strips.append(s)

    all_output = filter_chain.batch(all_input_for_filter)

    for s,r in zip(all_strips,all_output):
        if r:
            kept_strips.append(s)
    
    refined_context = "\n\n".join(kept_strips)

    return {
        'kept_strips':kept_strips,
        'refined_context':refined_context,
        "all_strips":all_strips

    }
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ('system',"Answer only from the context. If not in contex, say you don't know",),
        ('human', "Question : {question}\n\nContext:\n{context}")
    ]
)

generate_chain = answer_prompt | model | StrOutputParser()

def generate(state):
    question = state['question']
    context = state['refined_context']
    answer = generate_chain.invoke({'question':question,'context':context})
    return {'ans':answer}

def fail_node(state:State) ->State:
    return {
        'answer':f'FAIL :{state['reason']}'
    }

def ambiguous_node(state:State) ->State:
    return { "answer": f'amviguous:{state['reason']}'} 

def route_after_eval(state:State) -> str:
    if state['verdict'] == "CORRECT":
        return 'refine'
    elif state['verdict'] == "INCORRECT":
        return 'WEB_search'
    else:
        return "ambiguous"



g = StateGraph(State)
g.add_node("retrieve", retrieve_node)
g.add_node('eval_each_doc',doc_eval_node)
g.add_node("refine", refine_node)
g.add_node("generate", generate)
g.add_node('fail',fail_node)
g.add_node('ambiguous',ambiguous_node)

g.add_edge(START, "retrieve")
g.add_edge("retrieve", "eval_each_doc")

g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {"refine": "refine", "web_search": "fail", "ambiguous": "ambiguous"}
)
g.add_edge("refine", "generate")
g.add_edge("generate", END)
g.add_edge("fail", END)

app = g.compile()
print("Graph compiled successfully!")

res = app.invoke({
    "question": "What is Neural Networks?",

})

print("Answer:", res["ans"])
print("Kept sentences count:", len(res['kept_strips']))

a = 'all_strips'
print(len(res[a]))
print(res[a])
print(len(res[a]))

