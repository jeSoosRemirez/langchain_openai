from fastapi import Depends, FastAPI
from fastapi.security.api_key import APIKey

import auth
from ai_part import get_loader, qa_prompt, template, vector_store

app = FastAPI()


@app.get("/ask_question/{question}")
async def info(question: str, api_key: APIKey = Depends(auth.get_api_key)):
    """
    Returns answer of chat on the user question
    """
    # Tokens limit validation
    if len(template + question) > 4096:
        return {"error": "The question is too big"}

    document = get_loader()
    retriever = vector_store(document=document)
    result = qa_prompt(retriever=retriever, question=question)

    return {"result": result["result"]}


@app.get("/open")
async def info() -> dict:
    """
    Allowed without API key
    """
    return {"default variable": "Open Route"}
