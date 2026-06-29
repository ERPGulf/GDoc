from docling.document_converter import DocumentConverter
from gdoc.gdoc.models import dense_model_,sparse_model_,reranker_,tokenizer_
from gdoc.gdoc.parent_child_chunking import upsert_chunks
from gdoc.gdoc.flat_chunking import flat_chunk_upsert
from werkzeug.utils import secure_filename
import os 
import frappe

@frappe.whitelist(allow_guest = True)
def upload_doc():
    file = frappe.request.files.get("file")
    if not file:
        frappe.throw("File not found")
    file_name_ext = secure_filename(file.filename)
    file_name = os.path.splitext(file_name_ext)[0]
    file_type = os.path.splitext(file_name_ext)[1]
    if not file_name :
        frappe.throw("File name required.")
    file_doc = frappe.get_doc({
        "doctype":   "File",
        "file_name": file_name,
        "is_private": 1,
        "content":   file.read(),
    })
    file_doc.insert(ignore_permissions=True)
    gdoc = frappe.get_doc({
        "doctype": "GDOCs",
        "source": file_doc.file_url,
        "file_name": file_name,
        "file_type": file_type,
        "uploaded_by":   frappe.session.user,
        "changai_status": "Pending",
    })
    gdoc.insert(ignore_permissions = True)
    frappe.enqueue("gdoc.gdoc.upload_doc.chunk_router", queue='long', doc_id= gdoc.name,file_name = file_name, file_name_ext= file_name_ext)
    return {
    "doc_id":  gdoc.name,
    "status":  "Pending",
    "message": "File uploaded, indexing queued"
}

    
def chunk_router(doc_id, file_name, file_name_ext):
    converter = DocumentConverter()
    full_file_path = frappe.get_site_path("private","files",file_name_ext)
    result = converter.convert(full_file_path)
    doc = result.document
    extracted_text = doc.export_to_markdown()
    dense_model = dense_model_()
    tokenizer = tokenizer_()
    total_tokens   = len(tokenizer.encode(extracted_text))
    if total_tokens < 1500:
        try:
            flat_chunk_upsert(doc_id= doc_id,file_path = full_file_path,file_name = file_name)
            frappe.db.set_value('GDOCs', doc_id, 'changai_status', 'Completed')
            return {
                "message":"Document Flat Chunking is running in background"
            }
        except Exception as e:
            frappe.log_error(frappe.get_traceback(), f"chunk_router failed for {doc_id} and error is {str(e)}")

            return {
                "error":str(e)
            }
    else:
        try:
            upsert_chunks(doc_id= doc_id,file_path = full_file_path,file_name = file_name)
            frappe.db.set_value('GDOCs', doc_id, 'changai_status', 'Completed')
            return {
                "message":"Document Parent Child Chunking is running in background"
            }
        except Exception as e:
            frappe.log_error(frappe.get_traceback(), f"chunk_router failed for {doc_id} and error is {str(e)}")
            return {
                "error":str(e)
            }
