# ingest_documents.py

from document_reader import DocumentReader
from pathlib import Path
import time

if __name__ == "__main__":
    reader = DocumentReader(chroma_db_path="./chroma_db")
    unified_collection_name = "multimodal_data"

    base_data_dir = Path("/Users/snezhanakoleva/kg_rag/rag_system/macedonian_data")

    for file_path in base_data_dir.iterdir():
        if file_path.is_file(): 
            file_extension = file_path.suffix.lower()
             
        docs = reader.read_single_document(
            file_path=str(file_path)
        )
        
        if docs:
            start_time = time.time()
            reader.add_documents_to_collection(docs, collection_name=unified_collection_name)
            print(f"Successfully added {len(docs)} documents from {file_path.name} to '{unified_collection_name}'.")
            print(docs)

            end_time = time.time()
            actual_processing_time = end_time - start_time
            print(actual_processing_time)
        else:
            print(f"No documents generated for file: {file_path.name}.")
