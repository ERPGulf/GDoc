import json
import os
import frappe
from docling.document_converter import DocumentConverter
from gdoc.gdoc.clients import call_gemini
from typing import List,Optional


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
def get_meta(file_name_ext: Optional[str]):
    file_name = os.path.splitext(file_name_ext)[0]    
    custom_schema_structure ={}
    schema_template = get_schema(file_name,"invoice",custom_schema_structure)
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
        # first = extracted_text[:3000]
        # last = extracted_text[-1000:]
        # text = first + "\n\n" + last
        res = call_llm(full_text,schema_template)
        return full_text,res


def call_llm(text,schema):
    sys_prompt = f"""
    I will give you a extracted text from a document and also a template of metaschema.
    So your task is to fetch the relevant values for the scheam from the extracted text amd fill the ketysin the schema
    and final return the resultant schema afetr fetching values from extracted text.Must be accurtae result 
    no imagination allowed"""
    user_prompt =  f"""extracted text:{text} and schema is this {schema}.Must not imaginate.Give accurate only from extracted schema"""
    output = call_gemini(user_prompt,sys_prompt)
    return output

    

get_meta("PDF-A3_ACC-SINV-2026-00244_outputdfc11f.pdf")

def store_schema():
    pass
# get_schema("custom",{
#     "type": "object",
#     "title": "ZATCA Tax Invoice",
#     "properties": {
#         "invoice_type": {
#             "type": "string",
#             "enum": ["standard", "simplified"]
#         },
#         "invoice_number": {
#             "type": "string"
#         },
#         "invoice_date": {
#             "type": "string",
#             "format": "date"
#         },
#         "invoice_time": {
#             "type": "string"
#         },
#         "seller": {
#             "type": "object",
#             "properties": {
#                 "name": {"type": "string"},
#                 "address": {"type": "string"},
#                 "vat_number": {"type": "string"},
#                 "additional_ids": {
#                     "type": "array",
#                     "items": {"type": "string"}
#                 }
#             },
#             "required": ["name", "vat_number"]
#         },
#         "buyer": {
#             "type": "object",
#             "properties": {
#                 "name": {"type": "string"},
#                 "address": {"type": "string"},
#                 "vat_number": {"type": "string"}
#             },
#             "required": ["name"]
#         },
#         "line_items": {
#             "type": "array",
#             "items": {
#                 "type": "object",
#                 "properties": {
#                     "description": {"type": "string"},
#                     "quantity": {"type": "number"},
#                     "unit_price": {"type": "number"},
#                     "discount_amount": {"type": "number"},
#                     "tax_rate": {"type": "number"},
#                     "tax_amount": {"type": "number"},
#                     "line_total": {"type": "number"}
#                 },
#                 "required": ["description", "quantity", "unit_price", "line_total"]
#             }
#         },
#         "totals": {
#             "type": "object",
#             "properties": {
#                 "subtotal": {"type": "number"},
#                 "discount_total": {"type": "number"},
#                 "tax_total": {"type": "number"},
#                 "grand_total": {"type": "number"},
#                 "amount_due": {"type": "number"}
#             },
#             "required": ["grand_total", "tax_total"]
#         },
#         "payment": {
#             "type": "object",
#             "properties": {
#                 "payment_method": {"type": "string"},
#                 "paid_amount": {"type": "number"},
#                 "balance_due": {"type": "number"}
#             }
#         },
#         "zatca": {
#             "type": "object",
#             "properties": {
#                 "irn": {"type": "string"},
#                 "qr_code": {"type": "string"},
#                 "xml_hash": {"type": "string"},
#                 "currency": {"type": "string"}
#             }
#         }
#     },
#     "required": ["invoice_type", "invoice_number", "invoice_date", "seller", "buyer", "line_items", "totals"]
# })

