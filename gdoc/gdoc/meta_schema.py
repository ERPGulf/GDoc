import json
import os
import frappe
from gdoc.gdoc.clients import call_gemini
from typing import List,Optional
from gdoc.gdoc.models import dense_model_,sparse_model_,reranker_,tokenizer_,client_
from qdrant_client.models import PointStruct, SparseVector
import uuid
from gdoc.gdoc.ocr_extract_file import extract_text_from_file


def get_schema(doc_name: Optional[str], doc_type: str, custom_schema_structure: Optional[dict]):
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    if doc_type == "custom":
        schema_dir = os.path.join(data_dir, "schema", "custom_meta_schema")
        os.makedirs(schema_dir, exist_ok=True)
        file_path = os.path.join(schema_dir, f"{doc_name}.json")
        f = open(file_path, "w")
        json.dump(custom_schema_structure, f, indent=2)
        f.close()
        print(f"Saved to {file_path}")
        return custom_schema_structure
    else:
        file_path = os.path.join(data_dir, "schema", "standard_meta_schema", f"{doc_type}.json")
        f = open(file_path)
        data = json.loads(f.read())
        f.close()
        return data

max_chars = 4000
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend  # add this
@frappe.whitelist(allow_guest=True)
def get_meta(file_name_ext: Optional[str], doc_type: str, custom_schema_structure: dict = {}):
    if not file_name_ext:
        frappe.log_error("file_name_ext is missing", "get_meta: invalid input")
        return None

    file_name = os.path.splitext(file_name_ext)[0]
    file_ext = os.path.splitext(file_name_ext)[1]

    # --- get schema ---
    try:
        schema_template = get_schema(file_name, doc_type, custom_schema_structure)
        if schema_template is None:
            frappe.log_error(
                f"No schema found for file '{file_name}' with doc_type '{doc_type}'",
                "get_meta: schema not found"
            )
            return None
    except Exception as e:
        frappe.log_error(str(e)[:140], "get_meta: get_schema failed")
        return None

    # --- convert document to text ---
    try:
        full_file_path = frappe.get_site_path("private", "files", file_name_ext)
        full_text = extract_text_from_file(full_file_path,file_ext)
        if not full_text or not full_text.strip():
            frappe.log_error(
                f"No text extracted from '{file_name_ext}'",
                "get_meta: empty document"
            )
            return None
    except Exception as e:
        frappe.log_error(str(e)[:140], "get_meta: extraction failed")
        return str(e)

    # --- call LLM ---
    try:
        res = call_llm(full_text, schema_template)
        if not res:
            frappe.log_error("LLM returned empty response", "get_meta: call_llm failed")
            return None
        return res
    except Exception as e:
        frappe.log_error(str(e)[:140], "get_meta: call_llm failed")
        return None


def call_llm(text, schema):
    sys_prompt = """You are a strict information extraction engine. You will be given:
    1. Extracted text from a document.
    2. A JSON metaschema with field names, each having a "data_type" .
    YOUR TASK:
    - For every field in the schema, find the matching value from the extracted text and fill it into that field's "value" key.
    - Do NOT modify, add, or remove any keys, structure, or fields from the schema. Only fill in "value".
    - Do NOT guess, infer, calculate, or imagine any value that is not explicitly present in the extracted text.
    - If a field's value cannot be found in the text, set "value" to null. Do not leave it blank, do not put placeholder text, do not estimate.
    - Respect the declared "data_type" for each field:
    - "string" → return as text exactly as it appears (preserve formatting, don't reformat IDs/codes).
    - "float" → return as a number, not a string. Strip currency symbols/commas before converting.
    - "date" → return in DD-MM-YYYY format if the source date format can be unambiguously determined; otherwise null.
    - "list" → return an array of objects matching the nested schema structure, one object per matching item found in the text.
    - For "line_items" or similar nested lists, only include line items that are explicitly present in the text — do not pad, infer missing rows, or duplicate values across rows.
    - Additionally, generate "ai_tags": a list of short, relevant taxonomy tags (e.g. document type, category, key entities) based only on what is evident in the extracted text and the filled schema. Do not invent tags unrelated to the content.
    - If the metadata contains an identifier field representing the document's unique reference number in its source doctype (e.g. invoice_number for an Invoice, po_number for a Purchase Order similary for all other doctypes), extract that exact ID value as-is and return it separately as "erp_link_name" in the output — this must be the same raw string already present in the schema's value, not a new or reformatted ID.
    OUTPUT FORMAT — STRICT:
    Return ONLY a single valid JSON object, with no markdown formatting, no code fences, no explanations, no preamble, and no trailing commentary. The JSON must have exactly this top-level structure:
    {
    "meta": <the schema object, with all "value" fields filled in or null>,
    "ai_tags": [<string>, <string>, ...],
    "erp_link_name": <string or null>
    }
    If you cannot find any tags, return "ai_tags": [].
    If no identifier field is found, return "erp_link_name": null.
    Any deviation from this exact JSON structure is considered a failure.
    Return output in json .only no extra text or markdowns are allowed."""
    user_prompt = f"""Extracted text:
    {text}
    Schema to fill:
    {schema}
    Remember: fill values strictly from the extracted text above. No imagination, no estimation, no inferred values. Return ONLY the JSON object as specified."""

    output = call_gemini(user_prompt, sys_prompt)
    return output

def flatten_for_search(metadata: dict) -> str:
    parts = []
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, list):
            # line_items — extract all string/number values from each dict
            for entry in value:
                if isinstance(entry, dict):
                    parts.extend(str(v) for v in entry.values() if v)
                else:
                    parts.append(str(entry))
        
        elif isinstance(value, dict):
            parts.extend(str(v) for v in value.values() if v)
        
        else:
            parts.append(str(value))
    
    return " ".join(parts)

    
@frappe.whitelist(allow_guest=True)
def upsert_meta_doc(file_name_ext:str, custom_schema:dict={}):
    file_name = os.path.splitext(file_name_ext)[0]
    file_name = file_name.replace(" ","_")
    doc_id,doc_type= frappe.get_value("GDOCs",{'file_name': file_name},["name","doc_type"])
    res = get_meta(file_name_ext,doc_type,custom_schema)
    return res
    return text,img,full_text
    doc = frappe.get_doc("GDOCs",doc_id)
    try:
        if doc_type and frappe.db.exists("DocType", doc_type):
            doc.erp_link_doc = doc_type
        else:
            frappe.log_error(
                f"DocType '{doc_type}' does not exist in system",
                "upsert_meta_doc: invalid erp_link_doc"
            )
    except Exception as e:
        frappe.log_error(str(e)[:140], "upsert_meta_doc: erp_link_doc error")
    try:
        erp_link_name = res.get("erp_link_name")
        if erp_link_name and doc_type:
            if frappe.db.exists(doc_type, erp_link_name):
                doc.erp_link_name = erp_link_name
            else:
                frappe.log_error(
                    f"Record '{erp_link_name}' not found in '{doc_type}'",
                    "upsert_meta_doc: invalid erp_link_name"
                )
    except Exception as e:
        frappe.log_error(str(e)[:140], "upsert_meta_doc: erp_link_name error")
    try:
        tag_list = res.get("ai_tags", [])
        doc.set("ai_tags", [])
        for tag in tag_list:
            if not frappe.db.exists("Document Tag", tag):
                frappe.get_doc({
                    "doctype": "Document Tag",
                    "tag_name": tag
                }).insert(ignore_permissions=True)
            doc.append("ai_tags", {"tag": tag})
    except Exception as e:
        frappe.log_error(str(e)[:140], "upsert_meta_doc: ai_tags error")
    try:
        meta = res.get("meta")
        if meta:
            doc.metadata = json.dumps(meta,indent=2)
    except Exception as e:
        frappe.log_error(str(e)[:140], "upsert_meta_doc: metadata error")
        return str(e)

    doc.save(ignore_permissions=True)
    # try:
    #     searchable_texts = flatten_for_search(meta)
    #     sparse_model = sparse_model_()
    #     sparse_vectors = list(sparse_model.embed([searchable_texts]))[0]
    #     point = PointStruct(
    #         id=str(uuid.uuid4()),
    #         vector={
    #             "sparse": SparseVector(
    #                 indices=sparse_vectors.indices.tolist(),
    #                 values=sparse_vectors.values.tolist(),
    #             )
    #         },
    #         payload={
    #             "doc_id": doc_id,
    #             "doc_type": doc_type,
    #             **meta
    #         }
    #     )
    #     client = client_()
    #     client.upsert(collection_name="metadata_docs", points=[point])
    # except Exception as e:
    #     frappe.log_error(str(e)[:140],"upsert_meta_doc: qdrant upsert error")
    #     return str(e)

    return doc_id

#phonetic search in qdrant with rapidfuzz and jellyfish
def call_phonetic_fuzzy_search(word, top_k:int=5):
    client= client_()
    query_key = jellyfish.metaphone(word)
    must_filters = [
        FieldCondition(key="phonetic_key", match=MatchValue(value=query_key))
    ]
    results = client.scroll(
        collection_name="metadata_docs",
        scroll_filter=Filter(must=must_filters),
        limit=10 # pull candidates from the bucket
    )[0]
    scored = []
    for r in results:
        value = r.payload["value"]
        score = fuzz.ratio(word,value)
        scored.append((score,r.payload))
    scored.sort(key=lambda x:x[0], reverse=True)
    return scored[:top_k]

def call_bm25_search(word,top_k):
    sparse_query_result = list(sparse_model.query_embed(word))[0]
    sparse_query_vector = SparseVector(
        indices = sparse_query_result.indices.tolist(),
        values  = sparse_query_result.values.tolist()
    )
    results = client.query_points(
    collection_name = "metadata_docs",
    prefetch = [
        Prefetch(
            query  = sparse_query_vector,
            using  = "sparse",             # matches create_collection name
            limit  = 10
        )
    ],
    with_payload = True
    )
    hits = results.points
    return hits



def meta_search(query:str,entity_list:list=[]):
    #here entity_list is a list of entities and their types.
    res= []
    for item in entity_list:
        if item["type"]=="name_string":
            res.append(call_phonetic_fuzzy_search(item["name"],5))
        elif item["type"]=="id_string":
            res.append(call_bm25_search(item["name"],5))
    return res
    # if possible store site url also in both payload so
    # that after final reranking we can return these citaions to user


#for matching ids,codes,numbers only
def embed_exact_string(doc_id,doc_type,records):
    points=[]
    for item in records:
        sparse_query_result = list(sparse_model.query_embed(item["value"]))[0]
        sparse_query_vector = SparseVector(
            indices = sparse_query_result.indices.tolist(),
            values  = sparse_query_result.values.tolist()
        )
        point = PointStruct(
                id = str(uuid.uuid4()),
                vector = {
                    "sparse":SparseVector(
                        indices = sparse_vectors.indices.tolist(),
                        values  = sparse_vectors.values.tolist(),
                    )
                },
                payload = {
                    "doc_id":doc_id,
                    "doc_type":doc_type,
                    "field": item["key"],
                    "value": item["value"]
                    }
            )
        points.append(point)
    client=client_()
    client.upsert(collection_name = "metadata_docs",points=[point])
    return doc_id

# for matching text only
def fuzzy_phonetic(doc_id,doc_type,records):
    points=[]
    client =  client_()
    for item in records:
        value = item["value"]
        field = item["key"]
        if not value:
            continue
        first_word = value.split()[0]
        phonetic_match = jellyfish.metaphone(first_word)
        sparse_query_result = list(sparse_model.query_embed(value))[0]
        sparse_query_vector = SparseVector(
            indices=sparse_query_result.indices.tolist(),
            values=sparse_query_result.values.tolist())
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={"sparse": sparse_query_vector},
            payload = {
                "doc_id": doc_id,
                "doc_type": doc_type,
                "field": field,
                "value": value,
                "phonetic_key": phonetic_key 
            }
        )
    points.append(point)
    client.upsert(collection="metadata_docs",points=points)
    return doc_id







