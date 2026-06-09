# main.py
from fastapi import FastAPI

app = FastAPI()


@app.post("/upload")
async def upload_document():
    return {"message": "upload route works"}


@app.post("/ask")
async def ask_question():
    return {"message": "ask route works"}
