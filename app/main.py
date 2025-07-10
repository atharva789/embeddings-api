# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.embedder import embed_batch

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

app = FastAPI(
    title="ONNX Embedding Service",
    version="1.0"
)

@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    # embed_batch now returns the correct List[List[float]]
    embeddings_list = embed_batch(req.texts)
    
    # Directly return a dictionary that matches the EmbedResponse model
    return {"embeddings": embeddings_list}