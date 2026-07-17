from gdoc.gdoc.retrieval import RAGPipeline
import frappe
import os
from werkzeug.utils import secure_filename
# ---------- module-level singleton: models load once per worker, not per query ----------
_pipeline = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@frappe.whitelist()
def search(query: str):
    """API entry point — callable from frontend/Postman via
    /api/method/gdoc.gdoc.<this_module>.search"""
    return get_pipeline().ask(query)


@frappe.whitelist(allow_guest=True)
def upload_doc():
    results = []
    files = frappe.request.files.getlist("file")
    if not files:
        frappe.throw("Files not found")
    for file in files:
        try:
            file_name_ext = secure_filename(file.filename or "")
            file_name = os.path.splitext(file_name_ext)[0]
            file_type = os.path.splitext(file_name_ext)[1]
            if not file_name:
                results.append({"file_name": file.filename, "ok": False, "error": "invalid name"})
                continue
            file_doc = frappe.get_doc({
                "doctype":   "File",
                "file_name": file_name_ext,
                "is_private": 1,
                "content":   file.read(),
            })
            file_doc.insert(ignore_permissions=True)
            full_file_path = file_doc.get_full_path()
            gdoc = frappe.get_doc({
                "doctype": "GDOCs",
                "source": full_file_path,
                "file_name": file_name,
                "file_type": file_type,
                "uploaded_by":   frappe.session.user,
                "changai_status": "Pending",
                "assigned_to":""
            })
            gdoc.insert(ignore_permissions = True)
            frappe.enqueue("gdoc.gdoc.ingestion.chunk_router", queue='long',file_type = file_type, doc_id= gdoc.name,file_name = file_name, file_path = full_file_path)
            results.append({"file_name": file_name_ext, "doc_id": gdoc.name, "ok": True})
        except Exception as e:
            frappe.log_error(frappe.get_traceback(), f"upload failed: {file.filename}")
            results.append({"file_name": file_name_ext, "doc_id": gdoc.name, "ok": False})

    return results