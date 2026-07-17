from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from transformers import AutoTokenizer
from qdrant_client import QdrantClient
from gdoc.gdoc.onnx_embedder import ONNXEmbedder
import frappe

reranker = None
dense_model = None
sparse_model = None
tokenizer = None
client  = None

def client_():
    global client
    if client is None:
        doc = frappe.get_doc("Gdoc Settings")
        client = QdrantClient(url = doc.qdrant_url, api_key = doc.qdrant_api_key)
    return client


# def dense_model_():
#     global dense_model
#     if dense_model is None:
#         dense_model = ONNXEmbedder(
#             model_path = "/opt/hyrin/frappe-bench/apps/gdoc/gdoc/models/modernbert-int8",
#             model_file = "model_quantized.onnx"
#         )
#     return dense_model
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

dense_model = None
tokenizer = None


def dense_model_():
    global dense_model
    if dense_model is None:
        dense_model =  SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
    return dense_model


def tokenizer_():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "nomic-ai/nomic-embed-text-v2-moe",
            trust_remote_code=True
        )
    return tokenizer

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

# def tokenizer_():
#     global tokenizer
#     if tokenizer is None:
#         tokenizer = AutoTokenizer.from_pretrained(
#             "/opt/hyrin/frappe-bench/apps/gdoc/gdoc/models/modernbert-fp32",trust_remote_code=True
#         )
#     return tokenizer
