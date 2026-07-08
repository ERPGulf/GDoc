import os
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"

import io
import frappe
import pypdfium2
import pytesseract
import numpy as np
from PIL import Image
from typing import Optional
import fitz  # PyMuPDF
import pandas as pd
import json
import cv2
import pytesseract
import frappe
import numpy as np

def ocr_image(img):
    gray = get_grayscale(img)
    noise_removed = remove_noise(gray)
    thresh = thresholding(noise_removed)
    text = pytesseract.image_to_string(thresh)
    return text

def get_grayscale(page):
    img = pil_to_cv(page)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    
def remove_noise(image):
    return cv2.medianBlur(image,5)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def pil_to_cv(page):
    arr = np.array(page)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

import scipy.ndimage as ndi
def estimate_noise(gray):
    H, W = gray.shape
    M = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]])
    sigma = np.sum(np.abs(ndi.convolve(gray.astype(float), M)))
    return sigma * np.sqrt(0.5*np.pi) / (6.0*(W-2)*(H-2))

def choose_threshold_strategy(gray):
    # 1. Illumination uniformity — decides global vs adaptive
    bg = cv2.medianBlur(gray, 51)
    illum_std = bg.std()

    # 2. Contrast — decides whether to boost first
    p5, p95 = np.percentile(gray, [5, 95])
    contrast = p95 - p5

    if contrast < 60:                 # faded document
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

    if illum_std > 15:                # uneven lighting → adaptive
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=31, C=10)
    else:                             # uniform lighting → global Otsu
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def extract_scanned_pdf(pdf_path):
    from pdf2image import convert_from_path
    texts=[]
    pages = convert_from_path(pdf_path, 300)
    for i, page in enumerate(pages):
        gray = get_grayscale(page)
        noise_value = estimate_noise(gray)
        if noise_value > 10:
            gray = remove_noise(gray)
            cv2.imwrite(f"/opt/hyrin/frappe-bench/apps/gdoc/gdoc/gdoc/page_{i+1}_noise_removed_gray.png", gray)
        # convert to binary if the document lighting is uneven or faded
        binary = choose_threshold_strategy(gray)
        # cv2.imwrite(f"/opt/hyrin/frappe-bench/apps/gdoc/gdoc/gdoc/page_{i+1}_binary.png", binary)
        text = pytesseract.image_to_string(binary)
        texts.append(text)
    return texts

def extract_tables(page):
    tables = page.find_tables()         # auto-detect tables[web:16][web:19]
    table_elems = []

    for idx, t in enumerate(tables, start=1):
        df = t.to_pandas()              # structured rows / columns[web:16][web:19]
        # drop empty rows / cols
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

        if df.empty:
            continue

        table_elems.append({
            "type": "table",
            "id": f"table_{page.number+1}_{idx}",
            "bbox": list(t.bbox),       # (x0, y0, x1, y1)[web:14]
            "df": df
        })
    return table_elems


def extract_text_blocks(page, table_bboxes, margin=2):
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)[web:4]
    text_elems = []

    def overlaps_table(bbox):
        x0, y0, x1, y1 = bbox
        for tb in table_bboxes:
            tx0, ty0, tx1, ty1 = tb
            # simple overlap test with a small margin[web:15]
            if not (x1 < tx0 - margin or x0 > tx1 + margin or
                    y1 < ty0 - margin or y0 > ty1 + margin):
                return True
        return False

    for idx, b in enumerate(blocks, start=1):
        x0, y0, x1, y1, text, _, block_type = b
        if block_type != 0:             # skip images etc.[web:4]
            continue
        if not text.strip():
            continue
        if overlaps_table((x0, y0, x1, y1)):
            # this block is inside / overlapping a table; skip from narrative text[web:15]
            continue

        clean_text = " ".join(text.split()).strip()
        text_elems.append({
            "type": "text",
            "id": f"text_{page.number+1}_{idx}",
            "bbox": [x0, y0, x1, y1],
            "text": clean_text
        })

    return text_elems

def extract_digital_pdf_elements(filename):
    out = []
    with pymupdf.open(filename) as doc:
        for page in doc:
            tables = extract_tables(page)
            table_bboxes = [t["bbox"] for t in tables]
            texts = extract_text_blocks(page, table_bboxes)

            # Convert tables’ dataframes to serializable structures
            for t in tables:
                df = t["df"]
                df = df.replace(r'^\s*$', None, regex=True)
                df.dropna(axis=1, how="all")
                df = df.fillna("")
                out.append({
                    "type": "table",
                    "page": page.number + 1,
                    "bbox": t["bbox"],
                    "rows": df.values.tolist(),
                    "columns": df.columns.tolist()
                })

            for t in texts:
                out.append({
                    "type": "text",
                    # "id": t["id"],
                    "page": page.number + 1,
                    "bbox": t["bbox"],
                    "text": t["text"]
                })
    # sort by page, then vertical position (y0) for better reading order[web:15]
    out.sort(key=lambda e: (e["page"], e["bbox"][1]))
    for elem in out:
        elem.pop("bbox", None) 
    return out


def _preprocess_for_ocr(pil_image: Image.Image) -> Image.Image:
    """
    Preprocess image before OCR:
    1. upscale  — small text below Tesseract threshold gets ignored
    2. grayscale
    3. denoise
    4. adaptive threshold — handles uneven lighting universally
    5. deskew   — fixes tilted scans
    """
    try:
        import cv2

        width, height = pil_image.size
        if height < 3500:
            scale     = 3500 / height
            pil_image = pil_image.resize(
                (int(width * scale), int(height * scale)),
                Image.LANCZOS
            )

        img      = np.array(pil_image.convert("RGB"))
        img      = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        binary   = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )

        coords = np.column_stack(np.where(binary < 128))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:
                (h, w) = binary.shape
                center = (w // 2, h // 2)
                M      = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(
                    binary, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )

        return Image.fromarray(binary)

    except Exception as e:
        frappe.log_error(str(e)[:140], "_preprocess_for_ocr failed")
        return pil_image


def _upscale_small_blocks(pil_image: Image.Image) -> Image.Image:
    """
    Detects text blocks significantly smaller than the page average
    and upscales only those blocks so Tesseract doesn't ignore them.
    Works on any document layout — no hardcoding of positions.
    Used for minority regions like headers, labels, isolated text.
    """
    try:
        import cv2

        img           = np.array(pil_image)
        height, width = img.shape[:2]

        # dilate to merge nearby characters into detectable blocks
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        dilated = cv2.dilate(
            img if len(img.shape) == 2
            else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
            kernel, iterations=2
        )

        contours, _ = cv2.findContours(
            cv2.bitwise_not(dilated),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # collect valid block heights to find average
        block_heights = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 50 and h > 10:
                block_heights.append(h)

        if not block_heights:
            return pil_image

        avg_height = np.mean(block_heights)
        result     = Image.fromarray(img)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 50 or h < 10:
                continue  # skip noise

            # block is less than 60% of average height = small/minority block
            if h < avg_height * 0.6:
                block    = pil_image.crop((x, y, x + w, y + h))
                upscaled = block.resize((w * 2, h * 2), Image.LANCZOS)

                # paste upscaled block back at original position
                paste_x = max(0, x - w // 2)
                paste_y = max(0, y - h // 2)
                result.paste(upscaled, (paste_x, paste_y))

        return result

    except Exception as e:
        frappe.log_error(str(e)[:140], "_upscale_small_blocks failed")
        return pil_image


def _ocr_image(pil_image: Image.Image, lang: str = "eng+ara") -> str:
    """
    Two-pass OCR — guarantees minority regions are not ignored:
    Pass 1 PSM 6  — uniform block reading, good for dominant content
    Pass 2 PSM 11 — sparse text mode, finds text ANYWHERE on page
    Deduplicates and merges both results.
    """
    try:
        results = []

        text_psm6 = pytesseract.image_to_string(
            pil_image, lang=lang, config="--psm 6 --oem 1"
        ).strip()
        if text_psm6:
            results.append(text_psm6)

        text_psm11 = pytesseract.image_to_string(
            pil_image, lang=lang, config="--psm 11 --oem 1"
        ).strip()
        if text_psm11:
            results.append(text_psm11)

        seen  = set()
        lines = []
        for text in results:
            for line in text.splitlines():
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    lines.append(line)

        return "\n".join(lines)

    except Exception as e:
        frappe.log_error(str(e)[:140], "_ocr_image failed")
        return ""


def _ocr_blocks(pil_image: Image.Image, lang: str = "eng+ara") -> str:
    """
    Block-level OCR using image_to_data.
    Detects every word individually by bounding box — no region skipped.
    Reassembles words into lines in reading order.
    """
    try:
        data = pytesseract.image_to_data(
            pil_image,
            lang=lang,
            config="--psm 11 --oem 1",
            output_type=pytesseract.Output.DICT
        )

        blocks = {}
        for i in range(len(data["text"])):
            word      = data["text"][i].strip()
            conf      = int(data["conf"][i])
            block_num = data["block_num"][i]
            line_num  = data["line_num"][i]

            if not word or conf < 30:
                continue

            if block_num not in blocks:
                blocks[block_num] = {}
            if line_num not in blocks[block_num]:
                blocks[block_num][line_num] = []
            blocks[block_num][line_num].append(word)

        if not blocks:
            return ""

        lines = []
        for block_num in sorted(blocks.keys()):
            for line_num in sorted(blocks[block_num].keys()):
                line = " ".join(blocks[block_num][line_num])
                if line:
                    lines.append(line)

        return "\n".join(lines)

    except Exception as e:
        frappe.log_error(str(e)[:140], "_ocr_blocks failed")
        return ""


def _extract_from_pil_image(pil_image: Image.Image, lang: str = "eng+ara") -> str:
    """
    Full text extraction combining two-pass OCR + block-level detection.
    Any text on the page will be caught by at least one method.
    """
    try:
        seen  = set()
        lines = []

        for text in [_ocr_image(pil_image, lang), _ocr_blocks(pil_image, lang)]:
            for line in text.splitlines():
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    lines.append(line)

        return "\n".join(lines)

    except Exception as e:
        frappe.log_error(str(e)[:140], "_extract_from_pil_image failed")
        return ""


def _extract_tables_from_image(pil_image: Image.Image, lang: str = "eng+ara") -> str:
    """
    Extract structured tables using pytesseract bounding boxes.
    Groups words into rows by vertical proximity, columns by x position.
    """
    try:
        import pandas as pd

        data = pytesseract.image_to_data(
            pil_image,
            lang=lang,
            config="--psm 6 --oem 1",
            output_type=pytesseract.Output.DICT
        )

        items = []
        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not word or conf < 30:
                continue
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            items.append((y + h / 2, x + w / 2, h, word))

        if not items:
            return ""

        items.sort(key=lambda x: x[0])
        row_threshold = np.median([item[2] for item in items]) * 0.8

        rows        = []
        current_row = [items[0]]
        for item in items[1:]:
            if abs(item[0] - current_row[-1][0]) <= row_threshold:
                current_row.append(item)
            else:
                rows.append(sorted(current_row, key=lambda x: x[1]))
                current_row = [item]
        rows.append(sorted(current_row, key=lambda x: x[1]))

        multi_col = [r for r in rows if len(r) > 1]
        if len(multi_col) < 2:
            return ""

        table_data = [[item[3] for item in row] for row in rows]
        max_cols   = max(len(row) for row in table_data)
        table_data = [
            row + [""] * (max_cols - len(row))
            for row in table_data
        ]

        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        return df.to_markdown(index=False)

    except Exception as e:
        frappe.log_error(str(e)[:140], "_extract_tables_from_image failed")
        return ""

def _is_useful_image(pil_image: Image.Image) -> bool:
    """Skip tiny images like icons, logos."""
    width, height = pil_image.size
    return width >= 100 and height >= 100


def _extract_from_image_bytes(image_bytes: bytes, lang: str = "eng+ara") -> str:
    """
    Full extraction from raw image bytes:
    preprocess → table extraction → full text extraction
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))

        if not _is_useful_image(pil_image):
            return ""

        pil_image = _preprocess_for_ocr(pil_image)
        parts     = []

        table_text = _extract_tables_from_image(pil_image, lang)
        if table_text:
            parts.append(table_text)

        text = _extract_from_pil_image(pil_image, lang)
        if text:
            parts.append(text)

        return "\n".join(parts)

    except Exception as e:
        frappe.log_error(str(e)[:140], "_extract_from_image_bytes failed")
        return ""


def _extract_from_image_file(full_file_path: str, lang: str = "eng+ara") -> str:
    """Handle JPG, PNG, TIFF, BMP, WEBP etc."""
    try:
        pil_image = Image.open(full_file_path)
        pil_image = _preprocess_for_ocr(pil_image)
        parts     = []

        table_text = _extract_tables_from_image(pil_image, lang)
        if table_text:
            parts.append(table_text)

        text = _extract_from_pil_image(pil_image, lang)
        if text:
            parts.append(text)

        return "\n".join(parts)

    except Exception as e:
        frappe.log_error(str(e)[:140], "_extract_from_image_file failed")
        return ""
def overlaps(b1, b2):
    x0, y0, x1, y1 = b1
    a0, b0, a1, b1 = b2

    return not (
        x1 <= a0 or
        x0 >= a1 or
        y1 <= b0 or
        y0 >= b1
    )


def overlaps_any_table(block_bbox, table_bboxes):
    return any(
        overlaps(block_bbox, table_bbox)
        for table_bbox in table_bboxes
    )
# Python code example using PyPDF2
import PyPDF2
@frappe.whitelist(allow_guest=True)
def is_scanned_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        texts=[]
        for page in reader.pages:
            text = page.extract_text()
            texts.append(text)
        if any(text is "" for text in texts):
            return True  # PDF page is scanned
    return False  # All pages contain extractable text


def _extract_from_pdf(full_file_path: str, lang: str = "eng+ara") -> str:
    import pymupdf
    from gdoc.gdoc.digital_pdf_extract import extract_digital_pdf_elements
    try:
        if is_scanned_pdf(full_file_path):
            return extract_scanned_pdf(full_file_path)
        else:
            return extract_digital_pdf_elements(full_file_path) 
    except Exception as e:
        frappe.log_error(str(e)[:140], "_extract_from_pdf failed")
        return str(e)


def _extract_from_docx(full_file_path: str, lang: str = "eng+ara") -> str:
    try:
        import docx
        import pandas as pd

        doc   = docx.Document(full_file_path)
        parts = []

        for element in doc.element.body:
            tag = element.tag.split("}")[-1]

            if tag == "p":
                para = docx.text.paragraph.Paragraph(element, doc)
                text = para.text.strip()
                if text:
                    parts.append(text)

            elif tag == "tbl":
                table = docx.table.Table(element, doc)
                rows  = [
                    [cell.text.strip() for cell in row.cells]
                    for row in table.rows
                ]
                if rows:
                    df = pd.DataFrame(rows[1:], columns=rows[0])
                    parts.append(df.to_markdown(index=False))

        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    content = _extract_from_image_bytes(
                        rel.target_part.blob, lang
                    )
                    if content:
                        parts.append(content)
                except Exception:
                    pass

        return "\n\n".join(parts)

    except Exception as e:
        frappe.log_error(str(e)[:140], "_extract_from_docx failed")
        return ""


def _extract_from_xlsx(full_file_path: str, lang: str = "eng+ara") -> str:
    try:
        import openpyxl
        import pandas as pd

        wb    = openpyxl.load_workbook(full_file_path, data_only=True)
        parts = []

        for sheet_name in wb.sheetnames:
            ws          = wb[sheet_name]
            sheet_parts = [f"## Sheet: {sheet_name}"]

            data = [
                [str(cell) if cell is not None else "" for cell in row]
                for row in ws.iter_rows(values_only=True)
                if any(cell is not None for cell in row)
            ]
            if data:
                df = pd.DataFrame(data[1:], columns=data[0]) \
                    if len(data) > 1 else pd.DataFrame(data)
                sheet_parts.append(df.to_markdown(index=False))

            for image in ws._images:
                try:
                    content = _extract_from_image_bytes(
                        image._data(), lang
                    )
                    if content:
                        sheet_parts.append(content)
                except Exception:
                    pass

            parts.append("\n\n".join(sheet_parts))

        return "\n\n".join(parts)

    except Exception as e:
        frappe.log_error(str(e)[:140], "_extract_from_xlsx failed")
        return ""


def _extract_from_pptx(full_file_path: str, lang: str = "eng+ara") -> str:
    try:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE
        import pandas as pd

        prs   = Presentation(full_file_path)
        parts = []

        for slide_no, slide in enumerate(prs.slides, 1):
            slide_parts = [f"## Slide {slide_no}"]

            for shape in slide.shapes:

                if shape.has_text_frame:
                    text = "\n".join(
                        para.text.strip()
                        for para in shape.text_frame.paragraphs
                        if para.text.strip()
                    )
                    if text:
                        slide_parts.append(text)

                if shape.has_table:
                    rows = [
                        [cell.text.strip() for cell in row.cells]
                        for row in shape.table.rows
                    ]
                    if rows:
                        df = pd.DataFrame(rows[1:], columns=rows[0])
                        slide_parts.append(df.to_markdown(index=False))

                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        content = _extract_from_image_bytes(
                            shape.image.blob, lang
                        )
                        if content:
                            slide_parts.append(content)
                    except Exception:
                        pass

            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    slide_parts.append(f"Notes: {notes}")

            parts.append("\n\n".join(slide_parts))

        return "\n\n".join(parts)

    except Exception as e:
        frappe.log_error(str(e)[:140], "_extract_from_pptx failed")
        return ""


@frappe.whitelist(allow_guest=True)
def _extract_from_office(full_file_path: str, lang: str = "eng+ara") -> str:
    file_ext = os.path.splitext(full_file_path)[1]
    ext = file_ext.lower()
    if ext in [".docx", ".doc"]:
        return _extract_from_docx(full_file_path, lang)
    elif ext in [".xlsx", ".xls"]:
        return _extract_from_xlsx(full_file_path, lang)
    elif ext in [".pptx", ".ppt"]:
        return _extract_from_pptx(full_file_path, lang)
    else:
        frappe.log_error(
            f"Unsupported office format: {ext}",
            "_extract_from_office: unsupported"
        )
        return ""


def _extract_from_text_file(full_file_path: str) -> str:
    try:
        with open(full_file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        frappe.log_error(str(e)[:140], "_extract_from_text_file failed")
        return ""


def extract_text_from_file(full_file_path: str, file_ext: str, lang: str = "eng+ara") -> str:
    """
    Production-grade local extraction router.
    All OCR done locally by Tesseract — no cloud, no data leaves server.

    PDF          → fitz (text/tables) + pypdfium2 render + two-pass Tesseract
    Images       → preprocess + two-pass Tesseract + block-level detection
    Word/Excel   → dedicated libraries + Tesseract for embedded images
    Text files   → direct read
    """

    PDF_FORMATS    = [".pdf"]
    IMAGE_FORMATS  = [".jpg", ".jpeg", ".png", ".tiff",
                      ".tif", ".bmp", ".webp", ".gif"]
    OFFICE_FORMATS = [".docx", ".xlsx", ".pptx", ".doc",
                      ".xls", ".ppt", ".odt", ".ods", ".odp"]
    TEXT_FORMATS   = [".txt", ".csv", ".json", ".xml",
                      ".html", ".htm", ".md"]

    ext = file_ext.lower()

    if ext in PDF_FORMATS:
        return _extract_from_pdf(full_file_path, lang)

    elif ext in IMAGE_FORMATS:
        return _extract_from_image_file(full_file_path, lang)

    elif ext in OFFICE_FORMATS:
        return _extract_from_office(full_file_path, ext, lang)

    elif ext in TEXT_FORMATS:
        return _extract_from_text_file(full_file_path)

    else:
        frappe.log_error(
            f"Unsupported file type: {ext}",
            "extract_text_from_file: unsupported format"
        )
        return ""
