from docling.chunking import HybridChunker
from typing import List,Optional
from docling.document_converter import DocumentConverter
from qdrant_client.models import PointStruct, SparseVector
from qdrant_client.models import Prefetch, FusionQuery, Fusion
import frappe
from gdoc.gdoc.models import dense_model_,sparse_model_,reranker_,tokenizer_,client_
from qdrant_client import QdrantClient


def flat_chunk_upsert(doc_id,file_path,file_name):
    tokenizer = None
    client = None
    sparse_model = None
    dense_model = None
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document
    frappe.db.set_value('GDOCs', doc_id, 'changai_status', 'Processing')
    tokenizer = tokenizer_()
    MAX_TOKENS = 512
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=True
    )
    chunks=list(chunker.chunk(doc))
    texts = [chunk.text for chunk in chunks]
    dense_model = dense_model_()
    sparse_model = sparse_model_()
    sparse_vectors = list(sparse_model.embed(texts))
    dense_vectors = dense_model.encode(texts, normalize_embeddings=True, batch_size=32)
    points = []
    client = client_()
    for i,chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id = i,
                vector = {
                    "dense" : dense_vectors[i].tolist(),
                    "sparse":SparseVector(
                        indices = sparse_vectors[i].indices.tolist(),
                        values  = sparse_vectors[i].values.tolist(),
                    )
                },
                payload = {
                        "text":chunk.text,
                        "meta": {
                            "headings": chunk.meta.headings or [],
                            "filename": chunk.meta.origin.filename if chunk.meta.origin else file_name,
                        },
                        "file_name": file_name,
                        "url": file_path,
                        "doc_id":doc_id,
                    }
            )
        )
    batch_size = 100
    frappe.db.set_value('GDOCs', doc_id, 'changai_status', 'Chunked')
    for i in range(0,len(points),batch_size):
        client.upsert(collection_name="small_docs", points=points[i:i+batch_size])
    return {"message": f"Upsert Successfull for Small Docs Collection"}

@frappe.whitelist(allow_guest = True)
def hybrid_search(query_text: str, collection_name: str):
    reranker = None
    tokenizer = None
    client = None
    sparse_model = None
    dense_model = None
    client = client_()

    if not query_text:
        frappe.throw("Question is required")
    dense_model = dense_model_()
    sparse_model = sparse_model_()
    dense_query_vector = dense_model.encode(query_text, normalize_embeddings=True).tolist()
    sparse_query_result = list(sparse_model.query_embed(query_text))[0]
    sparse_query_vector = SparseVector(
        indices = sparse_query_result.indices.tolist(),
        values  = sparse_query_result.values.tolist()
    )
    results = client.query_points(
    collection_name = collection_name,
    prefetch = [
        Prefetch(
            query  = dense_query_vector,   # raw vector not models.Document
            using  = "dense",              # matches create_collection name
            limit  = 20                    # cast wide net for reranker
        ),
        Prefetch(
            query  = sparse_query_vector,
            using  = "sparse",             # matches create_collection name
            limit  = 20
        )
    ],
    query        = FusionQuery(fusion=Fusion.RRF),
    limit        = 20,                     # get 20 for reranker to score
    with_payload = True
    )
    hits = results.points
    pairs = [(query_text,hit.payload["text"]) for hit in hits]
    reranker = reranker_()
    scores = reranker.predict(pairs)
    reranked = sorted(
    zip(hits, scores),
    key     = lambda x: x[1],
    reverse = True )[:5]
    if collection_name == "large_docs":
        pids = [hit.payload["pid"] for hit,score in reranked]
        parent_docs = []
        for pid in pids:
            parent_text = frappe.get_value("GDoc Parent Chunks", {"pid": pid}, "text")
            parent_docs.append(parent_text)
        return parent_docs
    else:
        top_hits = [hit.payload["text"] for hit, score in reranked]
        return top_hits




