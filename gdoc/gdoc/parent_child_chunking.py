# Parent Child Retrieval
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from gdoc.gdoc.models import dense_model_,sparse_model_,reranker_,tokenizer_,client_
from fastembed import SparseTextEmbedding, SparseEmbedding
from qdrant_client.models import PointStruct, SparseVector
import uuid
import frappe

def store_parent_chunk(parent_chunk: list):
    doc = frappe.new_doc("GDoc Parent Chunks")
    doc.text = parent_chunk["enriched_text"]
    doc.file_name = parent_chunk["file_name"]
    doc.parent_doc_id = parent_chunk["doc_id"]
    doc.url = parent_chunk["url"]
    doc.pid = parent_chunk["pid"]
    doc.insert(ignore_permissions=True)
    return

def chunk_doc(doc_id, file_path, file_name):
    tokenizer = None
    client = None
    sparse_model = None
    dense_model = None
    client = client_()
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document
    frappe.db.set_value('GDOCs', doc_id, 'changai_status', 'Processing')
    tokenizer = tokenizer_()
    parent_chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=800,
        merge_peers=True)
    chunks = parent_chunker.chunk(dl_doc=doc)
    parent_chunks = []
    child_chunks = []
    for i, chunk in enumerate(chunks):
        parent_id = str(uuid.uuid4())
        contextualized_text = parent_chunker.contextualize(chunk=chunk)
        heading_path = chunk.meta.headings or []
        section_title = heading_path[-1] if heading_path else file_name
        parent_chunks.append({
            "enriched_text":contextualized_text,
            "pid":parent_id,
            "text":chunk.text,
            "file_name":file_name,
            "doc_id":doc_id,
            "chunk_index":i,
            "section_title": section_title,
            "url":file_path
        })
        tokens = tokenizer.encode(contextualized_text)
        start = 0
        position = 0
        child_size = 150
        OVERLAP = 20
        while(start< len(tokens)):
            end = min(start+child_size,len(tokens))
            child_text =  tokenizer.decode(tokens[start:end],skip_special_tokens=True)
            child_chunks.append({
                "text":child_text,
                "pid":parent_id,
                "doc_id":doc_id,
                "file_name":file_name,
                "url":file_path,
                "cid":str(uuid.uuid4()),
                "chunk_index": i,
                "section_title": section_title,
                "child_position": position
            })
            position += 1
            start =  end - OVERLAP
    child_texts = [c["text"] for c in child_chunks]
    dense_model_=dense_model_()
    sparse_model_ = sparse_model_()
    sparse_vectors = list(sparse_model.embed(child_texts))
    child_vectors = dense_model.encode(child_texts,normalize_embeddings=True,batch_size=32)
    for idx, vector in enumerate(child_vectors):
        child_chunks[idx]["vector"] = vector.tolist()
    for idx, sparse_vec in enumerate(sparse_vectors):
        child_chunks[idx]["sparse_indices"] = sparse_vec.indices.tolist()
        child_chunks[idx]["sparse_values"]  = sparse_vec.values.tolist()
    for parent_chunk in parent_chunks:
        store_parent_chunk(parent_chunk)
    frappe.db.set_value('GDOCs', doc_id, 'changai_status', 'Chunked')
    return parent_chunks, child_chunks

def upsert_chunks(doc_id, file_path, file_name):
    parent_chunks, child_chunks = chunk_doc(doc_id,file_path,file_name)
    client =  QdrantClient(path="/opt/hyrin/frappe-bench/apps/gdoc/gdoc/qdrant/storage")
    points = []
    for chunk in child_chunks:
        points.append(
        PointStruct(
            id = chunk["cid"],
            vector = {
                "dense": chunk["vector"],
                "sparse": SparseVector(
                    indices = chunk["sparse_indices"],
                    values =  chunk["sparse_values"]
                )
            },
            payload = {
                "text": chunk["text"],
                "cid":chunk["cid"],
                "pid": chunk["pid"],
                "chunk_index":chunk["chunk_index"],
                "section_title": chunk["section_title"],
                "doc_id": chunk["doc_id"],
                "file_name": chunk["file_name"],
                "url": chunk["url"],
                "position" : chunk["child_position"]
            }
        )
        )
    batch_size = 100
    for i in range(0,len(points),batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name = "large_docs",
            points          = batch
        )
    return {"message": f"Total upserted: {len(points)} child chunks"}

    
    

    

