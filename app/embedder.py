import onnxruntime as rt
import numpy as np
from transformers import AutoTokenizer
from functools import lru_cache

@lru_cache(maxsize=1)
def get_session():
    model_path = "models/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    session = rt.InferenceSession(
        f"{model_path}/model.onnx",
        providers=["CPUExecutionProvider"]
    )
    return tokenizer, session

def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def embed_batch(texts: list[str]) -> list[list[float]]:
    tokenizer, session = get_session()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="np"
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    # Discover the actual ONNX output name
    output_name = session.get_outputs()[0].name  # typically 'output_0'

    # Run the ONNX model
    ort_outputs = session.run([output_name], ort_inputs)
    token_embeddings = ort_outputs[0]  # shape: (batch, seq_len, hidden_dim)

    # Mean-pool over token embeddings
    sentence_embeddings = mean_pooling(token_embeddings, inputs["attention_mask"])

    return sentence_embeddings.tolist()
