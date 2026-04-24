import pandas as pd
from PyPDF2 import PdfReader
import re


def clean_text(text):
    """
    Clean extracted text while keeping meaning for RAG.
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove line breaks
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Fix common broken words from PDF extraction
    text = text.replace("com pleted", "completed")
    text = text.replace("pro gramme", "programme")
    text = text.replace("Gov ernment", "Government")
    text = text.replace("Min istry", "Ministry")

    # Remove strange characters but keep useful punctuation/numbers
    text = re.sub(r"[^a-zA-Z0-9.,;:%₵$()\-\/\s]", " ", text)

    # Fix multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_pdf(file_path):
    """
    Load PDF and clean extracted text.
    """
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "

    return clean_text(text)


def load_csv(file_path):
    """
    Load and clean CSV dataset.
    """
    df = pd.read_csv(file_path)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Fill missing values
    df = df.fillna("")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean column names
    df.columns = [
        clean_text(col).lower().replace(" ", "_")
        for col in df.columns
    ]

    # Clean every text value in the dataframe
    for col in df.columns:
        df[col] = df[col].apply(clean_text)

    return df


def csv_to_text_chunks(df):
    """
    Convert cleaned CSV rows into readable text chunks.
    """
    chunks = []

    for _, row in df.iterrows():
        row_text = " | ".join(
            [f"{col}: {row[col]}" for col in df.columns]
        )
        chunks.append(row_text)

    return chunks