
import os 
import fitz

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

persist_directory = "student_knowledge_base/vectorstore"


def vectorize_documents(documents):

    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        collection_name="student_knowledge_base",
        persist_directory="student_knowledge_base/vectorstore"
    )

    vector_store.persist()
    print("Vectorstore count:", vector_store._collection.count())



def read_pdf_from_folder(folder_path):
    pdf_files = [ f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    for root, dirs, files in os.walk(folder_path):
        pdf_files = [f for f in files if f.lower().endswith('.pdf')]

        for pdf_file in pdf_files:
            pdf_path = os.path.join(root, pdf_file)
            print(f"\nReading : {pdf_path}")

            doc = fitz.open(pdf_path)
            documents = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                if text.strip():
                    chunks = text_splitter.split_text(text)
                    for i,chunk in enumerate(chunks):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={"source": pdf_file, "page": page_num, "chunk": i}
                        ))

            doc.close()

            if documents:
                vectorize_documents(documents)


if __name__ == "__main__":
    folder_path = r'C:\Users\abhin\Documents\rt-agentic-ai-cert-week2\week3\data'
    read_pdf_from_folder(folder_path)





