import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
# This class is used for converting model files into onnx without pytorch
# reducing size and increasing speed.
class ONNXEmbedder:
    def __init__(self, model_path, model_file="model_quantized.onnx"):
        self.tokenizer = AutoTokenizer.from_pretrained(
    "/opt/hyrin/frappe-bench/apps/gdoc/gdoc/models/modernbert-fp32",trust_remote_code=True
)
        self.session   = ort.InferenceSession(f"{model_path}/{model_file}")

    def encode(self, texts, normalize_embeddings=True, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch  = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding        = True,
                truncation     = True,
                max_length     = 512,
                return_tensors = "np"
            )
            outputs    = self.session.run(None, {
                "input_ids":      inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            })
            embeddings = outputs[1]
            if normalize_embeddings:
                norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    @property
    def tokenizer_obj(self):
        return self.tokenizer