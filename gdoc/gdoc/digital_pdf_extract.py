import pymupdf
import pandas as pd
import json


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

def extract_digital_pdf_elements(filename) -> str:
    out = []
    with pymupdf.open(filename) as doc:          # fitz, since that's what you import
        for page in doc:
            tables = extract_tables(page)
            table_bboxes = [t["bbox"] for t in tables]
            texts = extract_text_blocks(page, table_bboxes)

            for t in tables:
                df = t["df"]
                df = df.replace(r'^\s*$', None, regex=True)
                df = df.dropna(axis=1, how="all")     # ← assignment was missing
                df = df.fillna("")
                out.append({
                    "page": page.number + 1,
                    "y": t["bbox"][1],
                    "content": df.to_markdown(index=False),   # table → markdown text
                })

            for t in texts:
                out.append({
                    "page": page.number + 1,
                    "y": t["bbox"][1],
                    "content": t["text"],
                })

    out.sort(key=lambda e: (e["page"], e["y"]))       # keep reading order
    return "\n\n".join(e["content"] for e in out)