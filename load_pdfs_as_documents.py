import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain.schema import Document

def extract_text_and_ocr(pdf_path, ocr_lang='hin+san'):
    doc = fitz.open(pdf_path)
    text_by_page = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():
            final_text = text
        else:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            final_text = pytesseract.image_to_string(img, lang=ocr_lang)

        text_by_page.append((page_num + 1, final_text.strip()))

    return text_by_page

def load_pdfs_as_documents(directory, ocr_lang='hin+san'):
    documents = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            print(f"📄 Loading: {filename}")
            page_texts = extract_text_and_ocr(pdf_path, ocr_lang=ocr_lang)

            for page_num, content in page_texts:
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": filename,
                            "page_number": page_num
                        }
                    )
                    documents.append(doc)
    print(f"✅ Total Documents Loaded: {len(documents)}")
    return documents
