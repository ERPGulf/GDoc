from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from transformers import AutoTokenizer
from qdrant_client import QdrantClient
from gdoc.gdoc.onnx_embedder import ONNXEmbedder

reranker = None
dense_model = None
sparse_model = None
tokenizer = None
client  = None

def client_():
    global client
    if client is None:
        QDRANT_URL = "http://127.0.0.1:6333"
        QDRANT_API_KEY = "6e6206a947ae52b71e5b0ca3211409e96bb511a7e84c2a3cd4101b6289a0edfb"
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return client


def dense_model_():
    global dense_model
    if dense_model is None:
        dense_model = ONNXEmbedder(
            model_path = "/opt/hyrin/frappe-bench/apps/gdoc/gdoc/models/modernbert-int8",
            model_file = "model_quantized.onnx"
        )
    return dense_model

def sparse_model_():
    global sparse_model
    if sparse_model is None:
        sparse_model   = SparseTextEmbedding("Qdrant/bm25")
    return sparse_model

def reranker_():
    global reranker
    if reranker is None:
        reranker =  CrossEncoder("BAAI/bge-reranker-v2-m3")
    return reranker

def tokenizer_():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "/opt/hyrin/frappe-bench/apps/gdoc/gdoc/models/modernbert-fp32",trust_remote_code=True
        )
    return tokenizer
