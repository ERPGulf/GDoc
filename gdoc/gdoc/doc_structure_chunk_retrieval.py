from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
import frappe
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from docling.document_converter import DocumentConverter
from gdoc.gdoc.models import dense_model_,sparse_model_,reranker_,tokenizer_,client_
from gdoc.gdoc.ingestion import FlatChunker, ParentChildChunker
from gdoc.gdoc.clients import call_model
import os
# from langchain_ollama import OllamaEmbeddings
from typing import Any, Dict, List, Optional, Union

COLLECTION_NAME = "gdoc_chunks"          # single collection, as discussed
TOP_K = 10                           # final contexts sent to the LLM

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
        self.llm = ChatOllama(model="mistral:7b", temperature=0,base_url="http://your-ollama-host:11434")
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()

    # ---------- retrieval ----------

    def retrieve(self, query: str,doc_ids :List[str]) -> list[dict]:
        """Hybrid search -> rerank -> parent expansion. Returns [{'text','url'}]."""
        doc_filter = None
        if doc_ids:
            doc_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]
        )
        PREFETCH_LIMIT  = max(20, 10 * len(doc_ids)) if doc_ids else 50                      # wide net for the reranker
        dense_query_vector = self.dense_model.encode(
            f"search_query: {query}", normalize_embeddings=True
        )

        sparse_result = list(self.sparse_model.query_embed(query))[0]
        sparse_query_vector = SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )
        if dense_query_vector.ndim == 2:        # (1, 768) -> (768,)
            dense_query_vector = dense_query_vector[0]
        dense_query_vector = dense_query_vector.tolist()
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,           # FIX: was undefined
            prefetch=[
                Prefetch(query=dense_query_vector, using="dense", limit=PREFETCH_LIMIT, filter=doc_filter),
                Prefetch(query=sparse_query_vector, using="sparse", limit=PREFETCH_LIMIT, filter=doc_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=doc_filter,
            limit=PREFETCH_LIMIT,
            with_payload=True,
        )
        hits = results.points
        if not hits:
            return []

        # rerank
        pairs = [(query, hit.payload["text"]) for hit in hits]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        # expand children to parents, dedupe parents by pid
        parent_groups = {}
        flat_hits = []
        for hit, _score in reranked:
            pid = hit.payload.get("pid")
            if pid:
                if pid not in parent_groups or _score > parent_groups[pid]:
                    parent_groups[pid] = _score
            else:
                flat_hits.append((hit, _score))
        candidates = [("parent", pid, s) for pid, s in parent_groups.items()] + \
             [("flat", hit, s) for hit, s in flat_hits]
        candidates.sort(key=lambda x: x[2], reverse=True)
        contexts = []
        for kind, ref, _s in candidates[:TOP_K]:
            if kind == "parent":
                parent = frappe.get_value("GDoc Parent Chunks", {"pid": ref},
                            ["text", "url"], as_dict=True)
                if parent:
                    contexts.append({"text": parent.text, "url": parent.url})
            else:
                contexts.append({"text": ref.payload["text"], "url": ref.payload.get("url")})
        return contexts

    # ---------- generation ----------

    @staticmethod
    def format_context(contexts: list[dict]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {c['url']}]\n{c['text']}" for c in contexts
        )

    def local_ask(self, query: str) -> dict:
        if not query or not query.strip():
            return {"answer": "Please provide a question.", "sources": []}
        doc = ""
        contexts = self.retrieve(query,doc)
        if not contexts:
            return {"answer": "I couldn't find anything relevant in the documents.",
                    "sources": []}

        answer = self.chain.invoke(
            {"question": query, "context": self.format_context(contexts)}
        )
        sources = list(dict.fromkeys(c["url"] for c in contexts if c.get("url")))
        return {"answer": answer, "sources": sources}
    def retrieve_debug(self, query: str, doc_ids: Optional[List[str]] = None, limit: int = 10) -> dict:
        """Compare retrieval strategies side by side. Debug only."""
        doc_filter = None
        if doc_ids:
            doc_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]
            )

        # --- encode once ---
        dense_vec = self.dense_model.encode(query, normalize_embeddings=True)
        if dense_vec.ndim == 2:
            dense_vec = dense_vec[0]
        dense_vec = dense_vec.tolist()

        sp = list(self.sparse_model.query_embed(query))[0]
        sparse_vec = SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())

        def fmt(points, scores=None):
            out = []
            for i, p in enumerate(points):
                out.append({
                    "score": float(scores[i]) if scores is not None else p.score,
                    "file": p.payload.get("file_name"),
                    "text": p.payload["text"][:120],       # preview only
                })
            return out

        # --- 1. dense only ---
        dense_res = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vec, using="dense",
            query_filter=doc_filter, limit=limit, with_payload=True,
        ).points

        # --- 2. sparse only ---
        sparse_res = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=sparse_vec, using="sparse",
            query_filter=doc_filter, limit=limit, with_payload=True,
        ).points

        # --- 3. hybrid RRF (no rerank) ---
        hybrid_res = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=limit * 3, filter=doc_filter),
                Prefetch(query=sparse_vec, using="sparse", limit=limit * 3, filter=doc_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=doc_filter, limit=limit, with_payload=True,
        ).points

        # --- 4. hybrid + rerank (your production path) ---
        pairs = [(query, h.payload["text"]) for h in hybrid_res]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(hybrid_res, scores), key=lambda x: x[1], reverse=True)
        rr_points = [h for h, _ in reranked]
        rr_scores = [s for _, s in reranked]

        return {
            "query": query,
            "dense_only": fmt(dense_res),
            "sparse_only": fmt(sparse_res),
            "hybrid_rrf": fmt(hybrid_res),
            "hybrid_reranked": fmt(rr_points, rr_scores),
        }
    def remote_ask(self, query: str,doc_ids:List[str] = None) -> dict:
        if not query or not query.strip():
            return {"answer": "Please provide a question.", "sources": []}

        contexts = self.retrieve(query,doc_ids)
        if not contexts:
            return {"answer": "I couldn't find anything relevant in the documents.",
                    "sources": []}
        self.dicision_system_prompt = """You are a helpful document assistant. Answer the user's question using ONLY the provided context from their documents.
            Guidelines:
            - Write the answer as natural, conversational language — full sentences a person would speak or write.
            - Do NOT return data structures, lists of key-value pairs, Python dicts, or JSON inside the answer. The answer must read like a human explanation, not like raw data.
            - If the question is about specific values (a price, a quantity, a total), state them in a sentence. Example: say "The unit price is $5.00 and the quantity listed is -10." — do NOT say "[{'unit_price': '$5.00', 'quantity': '-10'}]".
            - If a value looks inconsistent or missing in the document, mention that plainly in the sentence rather than omitting it.
            - Do not mention the words "context", "chunks", or "documents" — just answer naturally.

            OUTPUT FORMAT — STRICT:
            Return ONLY a single valid JSON object, with no markdown, no code fences, no preamble, and no trailing commentary.
            The JSON must have exactly this structure, where the value of "answer" is a natural-language string:
            {
            "answer": "A natural sentence or short paragraph answering the question."
            }"""

        self.dicision_prompt = """User Question:
        {query}

        Document chunks:
        {contexts}

        Answer the question in natural, conversational language based on the chunks above.
        Put your full answer as a single readable string in the "answer" field — never as a list or data object."""
        user_prompt = self.dicision_prompt.format(query=query,contexts=self.format_context(contexts))

        # answer = call_model(user_prompt,self.dicision_system_prompt)
        # answer_text = answer.get("answer") if isinstance(answer, dict) else str(answer)
        sources = list(dict.fromkeys(c["url"] for c in contexts if c.get("url")))
        return {"context":contexts
        # "answer": answer_text, "sources": sources
        }


@frappe.whitelist(allow_guest=True)  # nosemgrep: security.guest-whitelisted-method - intentional, validates credentials via OAuth client lookup and Frappe password grant before returning a token
def generate_token_secure(api_key: str, api_secret: str, app_key: str):
    try:
        try:
            app_key = base64.b64decode(app_key).decode("utf-8")
        except Exception:
            return Response(
                json.dumps(
                    {"message": "Security Parameters are not valid", "user_count": 0}
                ),
                status=401,
                mimetype=APPLICATION_JSON,
            )
        doc = frappe.db.get_value(
            "OAuth Client",
            {"app_name": app_key},
            ["name", "client_id", "client_secret", "user"],
            as_dict=True
        )
        if not doc:
            frappe.local.response["http_status_code"] = 401
            return {"ok": False, "error": "OAuth client not found / invalid app_key"}
        if doc.client_id is None:
            return Response(
                json.dumps(
                    {"message": "Security Parameters are not valid", "user_count": 0}
                ),
                status=401,
                mimetype=APPLICATION_JSON,
            )
        url = (
            frappe.local.conf.host_name
            + "/api/method/frappe.integrations.oauth2.get_token"
        )
        payload = {
            "username": api_key,
            "password": api_secret,
            "grant_type": "password",
            "client_id": doc.client_id,
            "client_secret": doc.client_secret,
        }
        response = requests.request("POST", url, data=payload)
        if response.status_code == STATUS_200:
            result_data = json.loads(response.text)
            return Response(
                json.dumps({"data": result_data}),
                status=STATUS_200,
                mimetype=APPLICATION_JSON,
            )
        else:
            frappe.local.response.http_status_code = 401
            return json.loads(response.text)
    except Exception as e:
        return Response(
            json.dumps({"message":str(e), "user_count": 0}),
            status=500,
            mimetype=APPLICATION_JSON,
        )
    

@frappe.whitelist(allow_guest=True)
def searching(query: str, doc: str = None, rec: str = None):
    docs = None
    if doc and rec:
        docs = frappe.get_all("GDOCs",
            filters={"erp_link_doc": doc, "erp_link_name": rec},
            pluck="name")
        if not docs:
            return {"answer": "No documents are linked to this record.", "sources": []}
    obj = RAGPipeline()
    return obj.remote_ask(query, docs)     # ← list, one call, done






