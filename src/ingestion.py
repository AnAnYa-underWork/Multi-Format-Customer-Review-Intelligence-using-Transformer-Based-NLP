import os
import pandas as pd
import fitz  # PyMuPDF
from docx import Document

def read_csv(file_path):
    df = pd.read_csv(file_path)
    documents = []

    for _, row in df.iterrows():
        text = str(row.get("reviewText", ""))
        source = f"csv_row_{_}"

        documents.append({
            "source": source,
            "text": text,
            "rating": row.get("overall", None),
            "date": row.get("reviewTime", None),
            "product": row.get("asin", None)
        })

    return documents

def read_txt(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "source": file,
                "text": text
            })

    return documents

def read_docx(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".docx"):
            path = os.path.join(folder_path, file)
            doc = Document(path)

            text = "\n".join([para.text for para in doc.paragraphs])

            documents.append({
                "source": file,
                "text": text
            })

    return documents

def read_pdf(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            doc = fitz.open(path)

            text = ""
            for page in doc:
                text += page.get_text()

            documents.append({
                "source": file,
                "text": text
            })

    return documents

def load_all_data(base_path):
    all_documents = []

    # CSV
    csv_path = os.path.join(base_path, "reviews.csv")
    all_documents.extend(read_csv(csv_path))

    # TXT
    txt_path = os.path.join(base_path, "complaints")
    all_documents.extend(read_txt(txt_path))

    # DOCX
    docx_path = os.path.join(base_path, "summaries")
    all_documents.extend(read_docx(docx_path))

    # PDF
    pdf_path = os.path.join(base_path, "reports")
    all_documents.extend(read_pdf(pdf_path))

    return all_documents