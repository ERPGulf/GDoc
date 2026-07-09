from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
import frappe
from docling.document_converter import DocumentConverter
from gdoc.gdoc.models import dense_model_,sparse_model_,reranker_,tokenizer_,client_
from gdoc.gdoc.parent_child_chunking import upsert_chunks
from gdoc.gdoc.chunker import FlatChunker, ParentChildChunker
from werkzeug.utils import secure_filename
import os
# from langchain_ollama import OllamaEmbeddings

COLLECTION_NAME = "gdoc_chunks"          # single collection, as discussed
PREFETCH_LIMIT = 20                      # wide net for the reranker
TOP_K = 5                                # final contexts sent to the LLM

PROMPT_TEMPLATE = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know. Keep the answer concise.

Question: {question}

Context:
{context}

Answer:"""
# NOTE: no <s>[INST] tokens — ChatOllama applies Mistral's chat template itself.


class RAGPipeline:
    def __init__(self):
        self.dense_model = dense_model_()
        self.sparse_model = sparse_model_()
        self.client = client_()
        self.reranker = reranker_()
        self.llm = ChatOllama(model="mistral:7b", temperature=0)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()

    # ---------- retrieval ----------

    def retrieve(self, query: str) -> list[dict]:
        """Hybrid search -> rerank -> parent expansion. Returns [{'text','url'}]."""
        dense_query_vector = self.dense_model.encode(
            query, normalize_embeddings=True
        ).tolist()

        sparse_result = list(self.sparse_model.query_embed(query))[0]
        sparse_query_vector = SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,           # FIX: was undefined
            prefetch=[
                Prefetch(query=dense_query_vector, using="dense", limit=PREFETCH_LIMIT),
                Prefetch(query=sparse_query_vector, using="sparse", limit=PREFETCH_LIMIT),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=PREFETCH_LIMIT,
            with_payload=True,
        )
        hits = results.points
        if not hits:
            return []

        # rerank
        pairs = [(query, hit.payload["text"]) for hit in hits]   # FIX: query_text -> query
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:TOP_K]

        # expand children to parents, dedupe parents by pid
        contexts = []
        seen_pids = set()
        for hit, _score in reranked:
            payload = hit.payload
            if payload.get("chunking_type") == "parent_child":   # FIX: colon + one spelling
                pid = payload["pid"]
                if pid in seen_pids:                             # FIX: dedupe parents
                    continue
                seen_pids.add(pid)
                parent = frappe.get_value(
                    "GDoc Parent Chunks",
                    {"pid": pid},
                    ["text", "url"],                             # FIX: fields as a list
                    as_dict=True,
                )
                if parent:
                    contexts.append({"text": parent.text, "url": parent.url})
            else:
                contexts.append({"text": payload["text"], "url": payload.get("url")})
        return contexts

    # ---------- generation ----------

    @staticmethod
    def format_context(contexts: list[dict]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {c['url']}]\n{c['text']}" for c in contexts
        )

    def ask(self, query: str) -> dict:
        if not query or not query.strip():
            return {"answer": "Please provide a question.", "sources": []}

        contexts = self.retrieve(query)
        if not contexts:
            return {"answer": "I couldn't find anything relevant in the documents.",
                    "sources": []}

        answer = self.chain.invoke(
            {"question": query, "context": self.format_context(contexts)}
        )
        sources = list(dict.fromkeys(c["url"] for c in contexts if c.get("url")))
        return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    obj = RAGPipeline()
    response = obj.retrive("What is ai?")
    print(response)
