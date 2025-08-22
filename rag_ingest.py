import os
import fitz
import yaml

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from paths import APP_CONFIG_FPATH, VECTOR_DB_DIR
from helper import load_yaml_config


def vectorize_documents(documents, embeddings):
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="student_knowledge_base",
        persist_directory=VECTOR_DB_DIR,
    )
    vector_store.persist()
    print("✅ Vectorstore count:", vector_store._collection.count())


def read_pdf_from_folder(folder_path, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    for root, _, files in os.walk(folder_path):
        pdf_files = [f for f in files if f.lower().endswith(".pdf")]

        for pdf_file in pdf_files:
            pdf_path = os.path.join(root, pdf_file)
            print(f"\n📖 Reading: {pdf_path}")

            doc = fitz.open(pdf_path)
            documents = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                if text.strip():
                    chunks = text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        documents.append(
                            Document(
                                page_content=chunk,
                                metadata={"source": pdf_file, "page": page_num, "chunk": i},
                            )
                        )

            doc.close()

            if documents:
                vectorize_documents(documents, embeddings)


if __name__ == "__main__":
    # Load app config
    app_config = load_yaml_config(APP_CONFIG_FPATH)

    embeddings = HuggingFaceEmbeddings(
        model_name=app_config["embedding_model"]
    )

    data_folder = os.path.join(os.path.dirname(__file__), "data")
    read_pdf_from_folder(data_folder, embeddings)
