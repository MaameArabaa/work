import pandas as pd
from PyPDF2 import PdfReader
import re


def clean_text(text):
    """
    Clean messy PDF text
    """
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("  ", " ")

    # Fix common broken words
    text = text.replace("com pleted", "completed")
    text = text.replace("Pro gramme", "Programme")
    text = text.replace("Gov ernment", "Government")

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_pdf(file_path):
    """
    Load and clean PDF text
    """
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "

    text = clean_text(text)

    return text


def load_csv(file_path):
    """
    Load CSV file
    """
    df = pd.read_csv(file_path)
    df = df.fillna("")

    return df


def csv_to_text_chunks(df):
    """
    Convert each CSV row into readable text
    """
    chunks = []

    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        chunks.append(row_text)

    return chunks