from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from qdrant_client.models import PointStruct, SparseVector, Filter, FieldCondition, MatchValue
import frappe
import uuid
import os
from gdoc.gdoc.models import  dense_model_,tokenizer_,sparse_model_, client_
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gdoc.gdoc.extract_text import extract_text_from_file,push_aitags


class BaseChunker:
    def __init__(self, doc_id, file_path,file_type, file_name):
        self.doc_id = doc_id
        self.file_path = file_path
        self.file_name = file_name
        self.file_type = file_type

        self.converter = DocumentConverter()
        self.dense_model = dense_model_()
        self.tokenizer = tokenizer_()
        self.sparse_model = sparse_model_()  # e.g. Qdrant/bm25
        self.client = client_()

        self.MAX_TOKENS = 512          # FIX: was `self.MAX_TOKENS - 512` (minus, not equals)
        self.batch_size = 100
        self.collection = "gdoc_chunks"

    def run(self):
        raise NotImplementedError

    def delete_existing_points(self):
        """Make reprocessing idempotent: remove old vectors for this doc first."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=self.doc_id))]
            ),
        )

    def upsert_points(self, points):
        for i in range(0, len(points), self.batch_size):
            self.client.upsert(
                collection_name = self.collection,
                points=points[i:i + self.batch_size],
            )


class FlatChunker(BaseChunker):

    def run(self):
        frappe.db.set_value("GDOCs", self.doc_id, "changai_status", "Processing")
        text = extract_text_from_file(self.file_path,self.file_type)
        try:
            push_aitags(self.doc_id,text)
        except Exception as e:
            frappe.log_error(frappe.get_traceback(), f"push_aitags failed for {self.doc_id}")
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,                 # counts real tokens, not characters
            chunk_size=self.MAX_TOKENS,     # 512
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],   # try paragraph → line → sentence → word
        )
        chunks = splitter.split_text(text)

        # Embed the contextualized text (heading breadcrumbs) — retrieves better.
        texts = [f"search_document: {c}" for c in chunks] 
        with open("/opt/hyrin/frappe-bench/apps/gdoc/gdoc/gdoc/ragas_data_small_docs.txt", "w", encoding="utf-8") as f:
            for text in texts:
                chunk = {
                    "chunk": text # make sure when fetching this falt chunks take upto 5 or 10 in single call 
                    # with claude
                }
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        sparse_vectors = list(self.sparse_model.embed(texts))
        dense_vectors = self.dense_model.encode(
            texts, normalize_embeddings=True, batch_size=32
        )

        points = []
        for i, chunk in enumerate(texts):  # FIX: append now correctly inside the loop
            points.append(
                PointStruct(
                    # FIX: was `id = i` — sequential ids collide across documents.
                    # uuid5 is deterministic per (doc_id, index) so re-uploading the
                    # same doc overwrites its own points instead of duplicating.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.doc_id}-{i}")),
                    vector={
                        "dense": dense_vectors[i].tolist(),
                        "sparse": SparseVector(
                            indices=sparse_vectors[i].indices.tolist(),
                            values=sparse_vectors[i].values.tolist(),
                        ),
                    },
                    payload={
                        "text": chunk,
                        "chunking_type":"flat",
                        # "meta": {
                        #     # "headings": chunk.meta.headings or [],
                        #     "filename": self.filename
                        #     # if chunk.meta.origin
                        #     # else self.file_name,
                        # },
                        "file_name": self.file_name,
                        "url": self.file_path,
                        "doc_id": self.doc_id,
                        "chunk_index": i,
                    },
                )
            )

        self.delete_existing_points()
        self.upsert_points(points)

        frappe.db.set_value("GDOCs", self.doc_id, "index_status", "Chunked")
        doc = frappe.get_doc("GDOCs",self.doc_id)
        # doc.set("ai_tags", [])
        # for tag in self.ai_tags:
        #     if not frappe.db.exists("Document Tag", tag):
        #         frappe.get_doc({
        #             "doctype": "Document Tag",
        #             "tag_name": tag
        #         }).insert(ignore_permissions=True)
        #     doc.append("ai_tags", {"tag": tag})

        return {"message": f"Upserted {len(points)} chunks to {self.collection}"}


class ParentChildChunker(BaseChunker):
    PARENT_MAX_TOKENS = 800
    CHILD_SIZE = 256
    OVERLAP = 30

    def store_parent_chunk(self, parent_chunk: dict):  # FIX: type hint was list
        doc = frappe.new_doc("GDoc Parent Chunks")
        doc.text = parent_chunk["text"]
        doc.file_name = parent_chunk["file_name"]
        doc.parent_doc_id = parent_chunk["doc_id"]
        doc.url = parent_chunk["url"]
        doc.pid = parent_chunk["pid"]
        doc.insert(ignore_permissions=True)

    def chunk_doc(self):
        frappe.db.set_value("GDOCs", self.doc_id, "changai_status", "Processing")
        text = extract_text_from_file(self.file_path, self.file_type)
        try:
            push_aitags(self.doc_id, text)
        except Exception:
            frappe.log_error(frappe.get_traceback(), f"push_aitags failed for {self.doc_id}")

        # PARENT splitter (what you have)
        parent_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=self.PARENT_MAX_TOKENS,      # use 800, not MAX_TOKENS — makes the constant real
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # NEW: CHILD splitter — same class, smaller size
        child_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=self.CHILD_SIZE,             # 256
            chunk_overlap=self.OVERLAP,             # 30
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        parent_chunks, child_chunks = [], []

        for i, parent_text in enumerate(parent_splitter.split_text(text)):
            parent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.doc_id}-parent-{i}"))
            parent_chunks.append({
                "chunking_type": "parentchild",
                "pid": parent_id,
                "text": parent_text,
                "file_name": self.file_name,
                "doc_id": self.doc_id,
                "chunk_index": i,
                "url": self.file_path,
            })

            # children via recursive splitter — replaces the whole while-loop
            for position, child_text in enumerate(child_splitter.split_text(parent_text)):
                child_chunks.append({
                    "chunking_type": "parentchild",
                    "text": child_text,
                    "pid": parent_id,
                    "doc_id": self.doc_id,
                    "file_name": self.file_name,
                    "url": self.file_path,
                    "cid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.doc_id}-child-{i}-{position}")),
                    "chunk_index": i,
                    "child_position": position,
                })


        child_texts = [
            f"search_document: {c['text']}"
            for c in child_chunks
        ]

        sparse_vectors = list(self.sparse_model.embed(child_texts))
        dense_vectors = self.dense_model.encode(
            child_texts, normalize_embeddings=True, batch_size=32
        )

        for idx, vector in enumerate(dense_vectors):
            child_chunks[idx]["vector"] = vector.tolist()
        for idx, sparse_vec in enumerate(sparse_vectors):
            child_chunks[idx]["sparse_indices"] = sparse_vec.indices.tolist()
            child_chunks[idx]["sparse_values"] = sparse_vec.values.tolist()

        for parent_chunk in parent_chunks:
            self.store_parent_chunk(parent_chunk)  # FIX: was missing self.

        return parent_chunks, child_chunks

    def run(self):
        # FIX: was chunk_doc(self.doc_id, ...) — method takes no args, reads self.*
        parent_chunks, child_chunks = self.chunk_doc()

        points = []
        for chunk in child_chunks:
            points.append(
                PointStruct(
                    id=chunk["cid"],
                    vector={
                        "dense": chunk["vector"],
                        "sparse": SparseVector(
                            indices=chunk["sparse_indices"],
                            values=chunk["sparse_values"],
                        ),
                    },
                    payload={
                        "text": chunk["text"],
                        "cid": chunk["cid"],
                        "pid": chunk["pid"],
                        "chunk_index": chunk["chunk_index"],
                        # "section_title": chunk["section_title"],
                        "doc_id": chunk["doc_id"],
                        "file_name": chunk["file_name"],
                        "url": chunk["url"],
                        "position": chunk["child_position"],
                    },
                )
            )

        self.delete_existing_points()
        self.upsert_points(points)

        frappe.db.set_value("GDOCs", self.doc_id, "changai_status", "Chunked")
        doc = frappe.get_doc("GDOCs",self.doc_id)
        # doc.set("ai_tags", [])
        # if self.ai_tags:
        #     for tag in self.ai_tags:
        #         if not frappe.db.exists("Document Tag", tag):
        #             frappe.get_doc({
        #                 "doctype": "Document Tag",
        #                 "tag_name": tag
        #             }).insert(ignore_permissions=True)
        #         doc.append("ai_tags", {"tag": tag})

        return {"message": f"Total upserted: {len(points)} child chunks"}


 
def chunk_router(doc_id,file_type, file_name,file_path):
    try:
        converter = DocumentConverter()
        tokenizer = tokenizer_()
        full_file_path = file_path
        result = converter.convert(full_file_path)
        doc = result.document
        extracted_text = doc.export_to_markdown()
        # embed_model = OllamaEmbeddings(model = "mistral:7b")
        TOKEN_THRESHOLD = 1500
        total_tokens   = len(tokenizer.encode(extracted_text))
        chunker_cls = FlatChunker if total_tokens < TOKEN_THRESHOLD  else ParentChildChunker
        chunker_cls(doc_id, full_file_path,file_type, file_name).run()
        frappe.db.set_value('GDOCs', doc_id, 'index_status', 'Completed')
    except Exception:
        frappe.log_error(frappe.get_traceback(), f"chunk_router failed for {doc_id}")
        frappe.db.set_value('GDOCs', doc_id, 'index_status', 'Failed')