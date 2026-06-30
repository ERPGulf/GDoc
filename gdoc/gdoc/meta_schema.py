import json
import os
import frappe
from docling.document_converter import DocumentConverter
from gdoc.gdoc.clients import call_gemini
from typing import List,Optional
from gdoc.gdoc.models import dense_model_,sparse_model_,reranker_,tokenizer_,client_
from qdrant_client.models import PointStruct, SparseVector
import uuid


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
#doc_type: str, custom_schema_structure: Optional[dict]
@frappe.whitelist(allow_guest=True)
def get_meta(file_name_ext: Optional[str],doc_type:str, custom_schema_structure:dict= {}):
    file_name = os.path.splitext(file_name_ext)[0]    
    custom_schema_structure ={}
    schema_template = get_schema(file_name,doc_type,custom_schema_structure)
    full_file_path = frappe.get_site_path("private","files",file_name_ext)
    if schema_template!=None:
        #ner extract
        converter = DocumentConverter()
        result = converter.convert(full_file_path)
        all_texts = []
        for text_item in result.document.texts:
            if text_item.text.strip():
                all_texts.append(text_item.text.strip())

        full_text = "\n".join(all_texts)
        res = call_llm(full_text,schema_template)
        data = json.load(res)
        return data


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
Any deviation from this exact JSON structure is considered a failure."""
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
    doc_id,doc_type= frappe.get_value("GDOCs",{'file_name': file_name},["name","doc_type"])
    res = get_meta(file_name_ext,doc_type,custom_schema)
    # need to add try except here for catching not existing doctypes
    frappe.set_value("GDOCs",doc_id,{"erp_link_doc":doc_type})
    doc = frappe.get_doc("GDOcs",doc_id)
    tag_list = res["ai_tags"]
    erp_link_name = res["erp_link_name"]
    # need to add try except here for catching not existing ids
    frappe.set_value("GDOCs",doc_id,{"erp_link_name":erp_link_name})
    for tag in tag_list:
        if not frappe.db.exists("Document Tag",{tag}):
            frappe.get_doc({
                "doctype":"Document Tag",
                "tag_name":tag
            }).insert(ignore_permissions=True)
        doc.append("ai_tags",{"tag": tag})
    doc.save(ignore_permissions=True)
    data  = res["meta"]
    try:
        meta = json.loads(data)
        frappe.db.set_value("GDOCs",doc_id, "metadata", data)
        return "Done"
    except Exception as e:
        return str(e)
    searchable_texts = flatten_for_search(meta)
    sparse_model = sparse_model_()
    sparse_vectors = list(sparse_model.embed([searchable_texts]))[0]
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
                    **meta
                    }
            )
    client = client_()
    client.upsert(collection_name = "metadata_docs",points=[point])
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







