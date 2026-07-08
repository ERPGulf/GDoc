from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from qdrant_client.models import PointStruct, SparseVector, Filter, FieldCondition, MatchValue
import frappe
import uuid

from gdoc.gdoc.models import dense_model_, sparse_model_, tokenizer_, client_


class BaseChunker:
    def __init__(self, doc_id, file_path, file_name):
        self.doc_id = doc_id
        self.file_path = file_path
        self.file_name = file_name

        self.converter = DocumentConverter()
        self.dense_model = dense_model_()
        self.tokenizer = tokenizer_()
        self.sparse_model = sparse_model_()  # e.g. Qdrant/bm25
        self.client = client_()

        self.MAX_TOKENS = 512          # FIX: was `self.MAX_TOKENS - 512` (minus, not equals)
        self.batch_size = 100
        self.collection = "gdoc"

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

        result = self.converter.convert(self.file_path)
        doc = result.document

        chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=self.MAX_TOKENS,
            merge_peers=True,
        )
        chunks = list(chunker.chunk(doc))

        # Embed the contextualized text (heading breadcrumbs) — retrieves better.
        texts = [chunker.contextualize(chunk=c) for c in chunks]
        sparse_vectors = list(self.sparse_model.embed(texts))
        dense_vectors = self.dense_model.encode(
            texts, normalize_embeddings=True, batch_size=32
        )

        points = []
        for i, chunk in enumerate(chunks):  # FIX: append now correctly inside the loop
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
                        "text": chunk.text,
                        "chunking_type":"flat",
                        "meta": {
                            "headings": chunk.meta.headings or [],
                            "filename": chunk.meta.origin.filename
                            if chunk.meta.origin
                            else self.file_name,
                        },
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

        return {"message": f"Upserted {len(points)} chunks to {self.collection}"}


class ParentChildChunker(BaseChunker):
    PARENT_MAX_TOKENS = 800
    CHILD_SIZE = 150
    OVERLAP = 20

    def store_parent_chunk(self, parent_chunk: dict):  # FIX: type hint was list
        doc = frappe.new_doc("GDoc Parent Chunks")
        doc.text = parent_chunk["enriched_text"]
        doc.file_name = parent_chunk["file_name"]
        doc.parent_doc_id = parent_chunk["doc_id"]
        doc.url = parent_chunk["url"]
        doc.pid = parent_chunk["pid"]
        doc.insert(ignore_permissions=True)

    def chunk_doc(self):
        frappe.db.set_value("GDOCs", self.doc_id, "changai_status", "Processing")

        result = self.converter.convert(self.file_path)  # FIX: self.file_path
        doc = result.document

        parent_chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=self.PARENT_MAX_TOKENS,
            merge_peers=True,
        )
        chunks = parent_chunker.chunk(dl_doc=doc)

        parent_chunks = []
        child_chunks = []

        for i, chunk in enumerate(chunks):
            parent_id = str(uuid.uuid4())
            contextualized_text = parent_chunker.contextualize(chunk=chunk)
            heading_path = chunk.meta.headings or []
            section_title = heading_path[-1] if heading_path else self.file_name

            parent_chunks.append(
                {
                    "chunking_type":"parentchild",
                    "enriched_text": contextualized_text,
                    "pid": parent_id,
                    "text": chunk.text,
                    "file_name": self.file_name,
                    "doc_id": self.doc_id,
                    "chunk_index": i,
                    "section_title": section_title,
                    "url": self.file_path,
                }
            )

            tokens = self.tokenizer.encode(contextualized_text)
            start = 0
            position = 0
            while start < len(tokens):
                end = min(start + self.CHILD_SIZE, len(tokens))
                child_text = self.tokenizer.decode(
                    tokens[start:end], skip_special_tokens=True
                )
                child_chunks.append(
                    {
                        "text": child_text,
                        "pid": parent_id,
                        "doc_id": self.doc_id,
                        "file_name": self.file_name,
                        "url": self.file_path,
                        "cid": str(uuid.uuid4()),
                        "chunk_index": i,
                        "section_title": section_title,
                        "child_position": position,
                    }
                )
                position += 1
                # FIX: infinite loop — when end hit len(tokens),
                # `start = end - OVERLAP` moved start BACKWARDS forever.
                if end == len(tokens):
                    break
                start = end - self.OVERLAP

        child_texts = [c["text"] for c in child_chunks]
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
                        "section_title": chunk["section_title"],
                        "doc_id": chunk["doc_id"],
                        "file_name": chunk["file_name"],
                        "url": chunk["url"],
                        "position": chunk["child_position"],
                    },
                )
            )

        self.delete_existing_points(self.collection)
        self.upsert_points(self.COLLECTION, points)

        frappe.db.set_value("GDOCs", self.doc_id, "changai_status", "Chunked")

        return {"message": f"Total upserted: {len(points)} child chunks"}


 
def chunk_router(doc_id, file_name, file_name_ext):
    try:
        converter = DocumentConverter()
        tokenizer = tokenizer_()
        full_file_path = frappe.get_site_path("private","files",file_name_ext)
        result = converter.convert(full_file_path)
        doc = result.document
        extracted_text = doc.export_to_markdown()
        # embed_model = OllamaEmbeddings(model = "mistral:7b")
        TOKEN_THRESHOLD = 1500
        total_tokens   = len(tokenizer.encode(extracted_text))
        chunker_cls = FlatChunker if total_tokens < TOKEN_THRESHOLD  else ParentChildChunker
        chunker_cls(doc_id, full_file_path, file_name).run()
        frappe.db.set_value('GDOCs', doc_id, 'index_status', 'Completed')
    except Exception:
        frappe.log_error(frappe.get_traceback(), f"chunk_router failed for {doc_id}")
        frappe.db.set_value('GDOCs', doc_id, 'index_status', 'Failed')
