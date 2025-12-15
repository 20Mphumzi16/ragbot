import os
import json
from load_documents import load_documents_from_folder
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


FOLDER_PATH = "documents"
DB_DIR = "./chrome_langchain_db"
COLLECTION_NAME = "grad_collection"
INDEX_FILE = "db_file_index.json"


def load_existing_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r") as f:
            return json.load(f)
    return {}


def save_index(index_data):
    with open(INDEX_FILE, "w") as f:
        json.dump(index_data, f, indent=4)


def database_needs_rebuild(new_index, old_index):
    return new_index != old_index


def rebuild_database(docs):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # Delete old DB directory
    if os.path.exists(DB_DIR):
        for root, dirs, files in os.walk(DB_DIR, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(DB_DIR)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    ids = [str(i) for i in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=ids)
    print("⚡ Vector database rebuilt successfully.")


def main():
    docs, new_index = load_documents_from_folder(FOLDER_PATH)
    old_index = load_existing_index()

    # Compare hashes
    if database_needs_rebuild(new_index, old_index):
        print("📌 Changes detected! Rebuilding vector DB...")
        rebuild_database(docs)
        save_index(new_index)
    else:
        print("✔ No changes detected. Vector DB is up to date.")


if __name__ == "__main__":
    main()
