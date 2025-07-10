from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from pathlib import Path

model_id = "sentence-transformers/all-MiniLM-L6-v2"
onnx_path = Path("models/all-MiniLM-L6-v2")

# Export model to ONNX
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save to desired path
model.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)
