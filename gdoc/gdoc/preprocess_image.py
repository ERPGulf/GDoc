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


@frappe.whitelist(allow_guest=True)
def extract_scanned_pdf(pdf_path):
    from pdf2image import convert_from_path
    texts=[]
    from gdoc.gdoc.ocr_extract_file import is_scanned_pdf
    if is_scanned_pdf(pdf_path):
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
    return False
