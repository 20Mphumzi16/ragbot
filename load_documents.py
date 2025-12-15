import os
import hashlib
from langchain_community.document_loaders import (
    TextLoader, CSVLoader, PDFPlumberLoader,
    Docx2txtLoader, JSONLoader, UnstructuredHTMLLoader, UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {
    ".txt", ".pdf", ".csv", ".docx", ".json", ".html", ".htm"
}


def compute_file_hash(filepath: str) -> str:
    """Returns a SHA256 hash of the file contents."""
    with open(filepath, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest()


def load_documents_from_folder(folder_path: str):
    docs = []
    file_hashes = {}

    # Walk through all subfolders recursively
    for root, _, files in os.walk(folder_path):
        for filename in files:

            # --- NEW: Skip MS Word temporary files ---
            if filename.startswith("~$"):
                print(f"Skipping temporary Word file: {filename}")
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                print(f"Skipping unsupported file: {filename}")
                continue

            full_path = os.path.join(root, filename)
            relative_path = os.path.relpath(full_path, folder_path)

            file_hashes[relative_path] = compute_file_hash(full_path)

            # Select loader
            if ext == ".txt":
                loader = TextLoader(full_path)

            elif ext == ".pdf":
                loader = PDFPlumberLoader(full_path)

            elif ext == ".csv":
                loader = CSVLoader(full_path)

            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(full_path)

            elif ext == ".json":
                loader = JSONLoader(full_path, jq_schema=".", text_content=False)

            elif ext in [".html", ".htm"]:
                loader = UnstructuredHTMLLoader(full_path)

            # Load & store metadata
            loaded_docs = loader.load()
            for d in loaded_docs:
                d.metadata["source_file"] = filename
                d.metadata["path"] = full_path
                d.metadata["relative_path"] = relative_path

            docs.extend(loaded_docs)

    # --- Chunking ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )

    chunked_docs = splitter.split_documents(docs)
    return chunked_docs, file_hashes
