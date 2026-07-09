import cv2
import pytesseract
import scipy.ndimage as ndi
from PIL import Image
import numpy as np
import frappe
# you must not have borders in the image for running this.
def _estimate_skew(gray: np.ndarray):
    """Robust skew estimate from near-horizontal Hough lines (text baselines).""" 
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=200,
        minLineLength=gray.shape[1] // 3, maxLineGap=20,
    )
    if lines is None:
        return None
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 15:            # keep only near-horizontal lines
            angles.append(a)
    if len(angles) < 5:
        return None
    return float(np.median(angles))

def get_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def maybe_invert(gray):
    if gray.mean() < 110: # means bg is dark than text
        return cv2.bitwise_not(gray), "inverted"
    return gray, "not_inverted"

def remove_noise(image):
    return cv2.medianBlur(image,3)

def pil_to_cv(page):
    arr = np.array(page)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def estimate_noise(gray):
    H, W= gray.shape
    M = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]])
    sigma = np.sum(np.abs(ndi.convolve(gray.astype(float), M)))
    return sigma * np.sqrt(0.5*np.pi) / (6.0*(W-2)*(H-2))

def maybe_binarize(gray):
    # 0. already binary? -> nothing to do
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    if (hist > hist.sum() * 0.01).sum() <= 2:
        return gray, "already_binary"

    # 1. faded/low contrast? -> boost first (kept even if we skip thresholding)
    p5, p95 = np.percentile(gray, [5, 95])
    if (p95 - p5) < 60:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # 2. uneven lighting? -> adaptive threshold (the one case we must handle)
    bg = cv2.medianBlur(gray, 51)          # background illumination map
    if bg.std() > 15:
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 10)
        return binary, "adaptive_binarize"

    # 3. even lighting -> DON'T binarize; Tesseract's internal Otsu handles it
    return gray, "left_to_tesseract"


def process_pil_image(pil, lang="eng+ara"):
    if pil.mode in("RGBA","LA","P"):
        rgba = pil.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask = rgba.split()[-1])
        pil = bg
    arr = np.array(pil.convert("RGB")) 
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) 
    gray, inv_tag = maybe_invert(gray)
    h, w = gray.shape
    if h < 1500:
        #upscale
        scale = 1500 / h
        gray = cv2.resize(gray, (int(w * scale), 1500),
                          interpolation=cv2.INTER_LANCZOS4)

    noise_value = estimate_noise(gray)
    if noise_value > 10:
        gray = remove_noise(gray)
    angle = _estimate_skew(gray)                  # 7
    if angle is not None and 1.0 < abs(angle) < 15.0:
        hh, ww= gray.shape
        M = cv2.getRotationMatrix2D((ww // 2, hh // 2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (ww, hh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    gray, tag = maybe_binarize(gray)
    parts     = []

    table_text = _extract_tables_from_image(gray, lang)
    if table_text:
        parts.append(table_text)

    text = _extract_from_pil_image(gray, lang)
    if text:
        parts.append(text)

    return "\n\n".join(parts)

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
def extract_image(full_file_path, lang):
    pil = Image.open(full_file_path)
    res = process_pil_image(pil,lang)
    # text = pytesseract.image_to_string(gray,lang=lang)
    return res
    
def extract_scanned_pdf(full_file_path, lang):
    parts = []
    from pdf2image import convert_from_path
    pages = convert_from_path(full_file_path, 300)
    for i, page in enumerate(pages):
        img = pil_to_cv(page)
        gray = get_grayscale(img)
        gray, tag = maybe_invert(gray)
        noise_value = estimate_noise(gray)
        if noise_value > 10:
            gray = remove_noise(gray)
        angle = _estimate_skew(gray)              # measure...
        if angle is not None and 1.0 < abs(angle) < 15.0:
            h, w = gray.shape                     # ...then rotate
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        binary,tag = maybe_binarize(gray)
        table_text = _extract_tables_from_image(binary, lang)
        if table_text:
            parts.append(table_text)

        text = _extract_from_pil_image(binary, lang)
        if text:
            parts.append(text)
    return  "\n\n".join(parts)

