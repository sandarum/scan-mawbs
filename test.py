from pdf2image import convert_from_path
from pypdf import PdfReader, PdfWriter
import pytesseract
import cv2
import numpy as np
import os
import re
from PIL import Image
from datetime import datetime

# -------- Directories -------- #
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -------- Log file created per run -------- #
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"run_{RUN_TIMESTAMP}.log")

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


# -------- OCR Configs -------- #
TESSERACT_CONFIGS = [
    "--oem 3 --psm 6 -c preserve_interword_spaces=1",
    "--oem 3 --psm 3 -c preserve_interword_spaces=1",
    "--oem 1 --psm 6 -c preserve_interword_spaces=1",
    "--oem 1 --psm 3 -c preserve_interword_spaces=1",
]

CARRIERS = ["QATAR AIRWAYS", "DHL AVIATION"]
PREFIXES = ["155", "157"]


# -------- MAWB Number Extraction -------- #
def extract_mawb_number(text, prefixes=PREFIXES):
    cleaned = text.upper().replace("MAWB", "")
    cleaned = re.sub(r"[|:;\n\r]", " ", cleaned)

    pattern = r"(" + "|".join(prefixes) + r")\D*(\d{6,10})"
    match = re.search(pattern, cleaned)
    if not match:
        return None

    prefix = match.group(1)
    digits = re.sub(r"\D", "", match.group(2))
    last_8 = digits[-8:]

    if len(last_8) != 8:
        return None

    return prefix + last_8


# -------- Preprocessing -------- #
def preprocess_deskew(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)

def preprocess_contrast(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def preprocess_simple_threshold(pil_image):
    img = np.array(pil_image)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(g, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def preprocess_orientation(pil_image):
    try:
        osd = pytesseract.image_to_osd(pil_image)
        rot = int(re.search(r"Rotate: (\d+)", osd).group(1))
        if rot != 0:
            return pil_image.rotate(-rot, expand=True)
    except:
        pass
    return pil_image

def preprocess_denoise(pil_image):
    img = np.array(pil_image)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = cv2.fastNlMeansDenoising(g, None, 10, 7, 21)
    return cv2.GaussianBlur(d, (3, 3), 0)

def preprocess_adaptive(pil_image):
    img = np.array(pil_image)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

def preprocess_morphology(pil_image):
    img = np.array(pil_image)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = cv2.threshold(g, 0, 255,
                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.dilate(t, np.ones((2, 2), np.uint8))


# -------- Utility -------- #
def crop_header(pil_image, fraction=0.3):
    w, h = pil_image.size
    return pil_image.crop((0, 0, w, int(h * fraction)))


def contains_airline(text):
    txt = text.upper()
    return any(c in txt for c in CARRIERS)


# -------- Fast Rotation Sweep -------- #
def try_rotations_fast(pil_image):
    small = pil_image.resize((pil_image.width // 2, pil_image.height // 2))
    for angle in [0, 90, 180, 270]:
        txt = pytesseract.image_to_string(
            small.rotate(angle, expand=True),
            config=TESSERACT_CONFIGS[0]
        )
        if contains_airline(txt):
            return True, f"rotate_{angle}"
    return False, None


# -------- Airline Page Detection -------- #
def detect_airline_page(pil_image):

    stages = [
        ("raw", lambda img: img),
        ("deskew", preprocess_deskew),
        ("contrast", preprocess_contrast),
        ("simple_threshold", preprocess_simple_threshold),
        ("orientation", preprocess_orientation),
        ("denoise", preprocess_denoise),
        ("adaptive_threshold", preprocess_adaptive),
        ("morphology", preprocess_morphology),
    ]

    for stage_name, func in stages:
        try:
            processed = func(pil_image)
            processed_img = (
                Image.fromarray(processed)
                if isinstance(processed, np.ndarray)
                else processed
            )

            for cfg in TESSERACT_CONFIGS:
                text = pytesseract.image_to_string(processed_img, config=cfg)
                if contains_airline(text):
                    return True, f"{stage_name} ({cfg})"

                header = crop_header(processed_img)
                text2 = pytesseract.image_to_string(header, config=cfg)
                if contains_airline(text2):
                    return True, f"{stage_name}_header ({cfg})"

        except Exception:
            continue

    matched, reason = try_rotations_fast(pil_image)
    if matched:
        return True, reason

    return False, None


# -------- Core Extraction -------- #
def extract_mawb_pages(pdf_path, output_dir):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    pages = convert_from_path(pdf_path, dpi=350)
    reader = PdfReader(pdf_path)
    matched_pages = 0
    stage_counts = {}
    total_pages = len(pages)

    for i, page in enumerate(pages):
        log(f"Processing page {i+1}/{total_pages}")
        matched, stage = detect_airline_page(page)

        if matched:
            matched_pages += 1
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            log(f"Detected MAWB (stage: {stage})")

            text_full = pytesseract.image_to_string(page, config="--psm 6")
            mawb_number = extract_mawb_number(text_full)

            # -------- Naming Logic -------- #
            if mawb_number:
                log(f"MAWB Number: {mawb_number}")
                out_file = os.path.join(output_dir, f"{mawb_number}.pdf")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"unknown_{filename}_p{i+1}_{timestamp}.pdf"
                out_file = os.path.join(output_dir, base_name)

                counter = 1
                while os.path.exists(out_file):
                    out_file = os.path.join(
                        output_dir,
                        f"unknown_{filename}_p{i+1}_{timestamp}_{counter}.pdf"
                    )
                    counter += 1

                log("MAWB number NOT found")

            writer = PdfWriter()
            writer.add_page(reader.pages[i])
            with open(out_file, "wb") as f:
                writer.write(f)

            log(f"Saved as: {os.path.basename(out_file)}")

        else:
            log("Not a MAWB page.")

    # Summary
    log(f"Summary: {matched_pages} of {total_pages} pages matched.")
    for stage, count in stage_counts.items():
        log(f"  - {stage}: {count}")


# -------- Entry Point -------- #
def process_pdfs():
    print("Scan-MAWBS OCR engine started...")
    print("Processing PDF files...\n")

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            log(f"Scanning file: {filename}")
            extract_mawb_pages(
                os.path.join(INPUT_DIR, filename),
                OUTPUT_DIR
            )

    print("Completed!")
    print(f"Log file created at: logs/{os.path.basename(LOG_FILE)}")


if __name__ == "__main__":
    process_pdfs()