import os
import json
import re 
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import pandas as pd
from bs4 import BeautifulSoup
import markdown
import chromadb
from chromadb.utils import embedding_functions
from .old_read_csv_file import read_csv_file 
import tiktoken 
import importlib.util
from .preprocess_pdf_files import split_text_into_chunks
from .old_document_classifier import DocumentClassifier

class DocumentReader:
    def __init__(self, chroma_db_path="./chroma_db2"):
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.supported_extensions = {'.txt', '.pdf', '.md', '.docx', '.html', '.csv', '.pptx', '.xlsx'} 
        self.default_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.classifier = DocumentClassifier()

    def info(self):
        """Prints information about the DocumentReader instance."""
        try:
            collection = self.client.get_or_create_collection(
                name="aws_case_data",
                embedding_function=self.default_embedding_function
            )
            count = collection.count()
            print(f"Collection '{collection.name}' has {count} documents.")
            if count > 0:
                results = collection.peek(limit=5)
                metadatas = results.get('metadatas', [])
                documents = results.get('documents', [])
                print("Sample Documents & Metadatas (first 5 documents):")
                for i in range(min(len(documents), len(metadatas))):
                    print(f"  Content (first 100 chars): {documents[i][:100]}...")
                    print(f"  Metadata: {metadatas[i]}")
            else:
                print("No documents in collection to display metadatas.")
        except Exception as e:
            print(f"Error accessing collection info: {e}")

    def _process_text_file(self, file_path: str, content: str, file_type: str, 
                          chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """Common method for processing text-based files."""
        # Classify document
        document_category = self.classifier.classify_document(file_path, content)
        
        # Use default chunking parameters if not provided
        chunk_size = chunk_size or 1500
        overlap = overlap or 200
        
        # Use the existing chunking function
        text_chunks = split_text_into_chunks(content, chunk_size=chunk_size, overlap=overlap)
        
        base_metadata = {
            'file_type': file_type,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path)
        }
        
        documents = []
        for i, chunk in enumerate(text_chunks):
            processed_chunk = chunk.lower().strip()
            if processed_chunk:
                metadata = {
                    **base_metadata,
                    'chunk_type': f'{file_type}_chunk',
                    'chunk_index': i,
                    'chunk_length': len(processed_chunk),
                    'num_words': len(processed_chunk.split())
                }
                enhanced_metadata = self.classifier.enhance_metadata(metadata, document_category, processed_chunk)
                documents.append({'content': processed_chunk, 'metadata': enhanced_metadata})
        
        return documents

    def read_text_file(self, file_path: str, chunk_size: int = None, overlap: int = None, 
                      token_based_chunking: bool = False) -> Union[List[Dict[str, Any]], None]:
        """Read and process text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Handle Project Gutenberg books
            start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
            end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
            start_index = content.find(start_marker)
            if start_index != -1:
                start_index = content.find('\n', start_index) + 1
                end_index = content.rfind(end_marker)
                content = content[start_index:end_index if end_index != -1 else len(content)]
            
            # Normalize whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            return self._process_text_file(file_path, content, 'txt', chunk_size, overlap)
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return None

    def read_markdown_file(self, file_path: str, chunk_size: int = None, overlap: int = None, 
                          token_based_chunking: bool = False) -> Union[List[Dict[str, Any]], None]:
        """Read and process markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, "html.parser")
            content = soup.get_text(separator=' ', strip=True)
            content = re.sub(r'\s+', ' ', content).strip()
            
            return self._process_text_file(file_path, content, 'markdown', chunk_size, overlap)
        except Exception as e:
            print(f"Error processing markdown file {file_path}: {e}")
            return None

    def read_html_file(self, file_path: str, chunk_size: int = None, overlap: int = None, 
                      token_based_chunking: bool = False) -> Union[List[Dict[str, Any]], None]:
        """Read and process HTML files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, "html.parser")
                content = soup.get_text(separator=' ', strip=True)
            
            content = re.sub(r'\s+', ' ', content).strip()
            documents = self._process_text_file(file_path, content, 'html', chunk_size, overlap)
            
            # Add HTML-specific metadata
            title_tag = soup.find('title')
            if title_tag and documents:
                title = title_tag.get_text(strip=True)
                for doc in documents:
                    doc['metadata']['title'] = title
            
            return documents
        except Exception as e:
            print(f"Error reading HTML {file_path}: {e}")
            return None

    def read_xlsx_file(self, file_path: str, chunk_size: int = None, overlap: int = None, 
                      token_based_chunking: bool = False) -> Union[List[Dict[str, Any]], None]:
        """Read and process Excel files."""
        try:
            xls = pd.ExcelFile(file_path)
            content = ""
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                content += f"\n--- Sheet: {sheet_name} ---\n{df.to_string(index=False)}\n"
            
            content = re.sub(r'\s+', ' ', content).strip()
            documents = self._process_text_file(file_path, content, 'xlsx', chunk_size, overlap)
            
            # Add Excel-specific metadata
            if documents:
                for doc in documents:
                    doc['metadata']['sheet_names'] = xls.sheet_names
            
            return documents
        except Exception as e:
            print(f"Error reading XLSX {file_path}: {e}")
            return None

    def _call_preprocess_script(self, script_path, function_name, file_path, chunk_size, overlap):
        """Process PDF, DOCX, PPTX files using external scripts."""
        spec = importlib.util.spec_from_file_location(function_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        preprocess_func = getattr(module, function_name)

        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        file_dict = {"name": os.path.basename(file_path), "content": file_bytes}
        
        result = preprocess_func(file_dict, chunk_size=chunk_size, overlap=overlap)
        if isinstance(result, dict) and 'result' in result:
            result = result['result']
        
        chunks = result.get('chunks', [])
        metadata = result.get('metadata', {})
        full_text = ' '.join(chunks) if chunks else ''
        
        # Classify and enhance documents
        document_category = self.classifier.classify_document(file_path, full_text)
        enhanced_documents = []
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                base_metadata = {**metadata, "chunk_index": i, 'chunk_length': len(chunk), 'num_words': len(chunk.split())}
                enhanced_metadata = self.classifier.enhance_metadata(base_metadata, document_category, chunk)
                enhanced_documents.append({"content": chunk.lower().strip(), "metadata": enhanced_metadata})
        
        return enhanced_documents

    def read_single_document(self, file_path: str, chunk_size: int = 1500, overlap: int = 200, 
                           token_based_chunking: bool = False, csv_content_columns: Optional[List[str]] = None,
                           csv_metadata_columns: Optional[List[str]] = None, csv_delimiter: str = ',',
                           csv_encoding: str = 'utf-8', csv_skip_empty_lines: bool = True) -> Union[List[Dict[str, Any]], None]:
        """Read a single document based on its file extension."""
        file_extension = Path(file_path).suffix.lower()
        
        # PDF, DOCX, PPTX files
        preprocess_map = {
            '.pdf': ("preprocess_pdf_files.py", "preprocess_pdf"),
            '.docx': ("preprocess_doc_files.py", "preprocess_doc"),
            '.pptx': ("preprocess_powerpoint_files.py", "preprocess_presentation"),
        }
        
        if file_extension in preprocess_map:
            script, func = preprocess_map[file_extension]
            script_path = os.path.join(os.path.dirname(__file__), script)
            return self._call_preprocess_script(script_path, func, file_path, chunk_size, overlap)
        
        # Text-based files
        text_readers = {
            '.txt': self.read_text_file,
            '.md': self.read_markdown_file,
            '.html': self.read_html_file,
            '.xlsx': self.read_xlsx_file
        }
        
        if file_extension in text_readers:
            return text_readers[file_extension](file_path, chunk_size, overlap, token_based_chunking)
        
        # CSV files
        elif file_extension == '.csv':
            return read_csv_file(file_path, csv_content_columns, csv_metadata_columns, 
                                csv_delimiter, csv_encoding, csv_skip_empty_lines)
        
        print(f"WARNING: Unsupported file type: {file_extension} for file: {file_path}")
        return None

    def scan_directory(self, directory_path: str, recursive: bool = True) -> List[str]:
        """Scan directory for supported document files."""
        document_files = []
        directory = Path(directory_path)

        if not directory.exists() or not directory.is_dir():
            print(f"Error: Directory not found: {directory_path}")
            return []

        pattern = directory.rglob("*") if recursive else directory.iterdir()
        for file_path in pattern:
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                document_files.append(str(file_path))

        return document_files

    def create_or_get_collection(self, collection_name: str):
        """Create or get a ChromaDB collection."""
        return self.client.get_or_create_collection(name=collection_name, embedding_function=self.default_embedding_function)

    def add_documents_to_collection(self,
                                      documents: List[Dict[str, Any]],
                                      collection_name: str = "my_documents",
                                      batch_size: int = 500) -> None: # Renamed chunk_size to batch_size
       
        if not documents:
            print(f"No documents provided to add to collection '{collection_name}'.")
            return
        print("Checkpoint 1")
        collection = self.create_or_get_collection(collection_name)
        total_added_count = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            ids_batch = []
            contents_batch = []
            metadatas_batch = []

            for j, doc in enumerate(batch):
                content = doc.get('content')
                metadata = doc.get('metadata')

                if not (isinstance(content, str) and content.strip() and isinstance(metadata, dict)):
                    print(f"WARNING: Skipping document {i + j} (batch index {j}) with missing or invalid 'content' or 'metadata'. Doc info: {doc.get('metadata', 'No Metadata')}")
                    continue

                # --- Robust ID Generation ---
                doc_id_prefix = metadata.get('file_type', 'unknown')
                file_name_part = os.path.splitext(metadata.get('file_name', 'no_file'))[0] # Get base name without extension

                # Use original_row_index for CSV, chunk_index for other text types
                # Fallback to general batch index for robustness
                source_identifier = ""
                if doc_id_prefix == 'csv' and 'original_row_index' in metadata:
                    source_identifier = f"row_{metadata['original_row_index']}"
                elif 'chunk_index' in metadata: # Covers all other chunked file types
                    source_identifier = f"chunk_{metadata['chunk_index']}"
                else:
                    source_identifier = f"part_{i + j}" # General fallback if no specific index

                doc_id = f"{doc_id_prefix}_{file_name_part}_{source_identifier}"
                ids_batch.append(doc_id)
                contents_batch.append(content)
                
                # --- Metadata Cleaning and Type Conversion for ChromaDB ---
                cleaned_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        cleaned_metadata[k] = v
                    elif v is None:
                        cleaned_metadata[k] = ""
                    else:
                        try:
                            cleaned_metadata[k] = json.dumps(v)
                        except TypeError:
                            print(f"WARNING: Metadata key '{k}' for doc {doc_id} has un-serializable type {type(v)}. Converting to string.")
                            cleaned_metadata[k] = str(v)

                metadatas_batch.append(cleaned_metadata)

            if ids_batch:
                try:
                    collection.add(
                        documents=contents_batch,
                        ids=ids_batch,
                        metadatas=metadatas_batch
                    )
                    total_added_count += len(ids_batch)
                    print(f"Added {len(ids_batch)} documents to collection '{collection_name}' in batch {i // batch_size + 1}. Current total: {total_added_count}")
                except Exception as e:
                    print(f"Error adding batch to collection '{collection_name}': {e}")
                    raise
            else:
                print(f"No valid documents found in batch {i // batch_size + 1} to add to collection '{collection_name}'.")

        print(f"Finished adding documents. Total documents added to '{collection_name}': {total_added_count}. Collection count: {collection.count()}")

    def process_directory(self, directory_path: str, collection_name: str = "my_documents",
                         recursive: bool = True, chunk_size: int = 1500, overlap: int = 200,
                         token_based_chunking: bool = False,
                         
                         csv_content_columns: Optional[List[str]] = None,
                         csv_metadata_columns: Optional[List[str]] = None,
                         csv_delimiter: str = ',',
                         csv_encoding: str = 'utf-8',
                         csv_skip_empty_lines: bool = True,
                         batch_size_chroma: int = 500 # Added for clarity with add_documents_to_collection
                         ) -> None:
        print(f"Scanning directory: {directory_path}")
        document_files = self.scan_directory(directory_path, recursive)
        print(f"Found {len(document_files)} supported files")

        documents_to_add = []
        for file_path in document_files:
            print(f"Processing: {file_path}")
            # Pass all relevant parameters to read_single_document
            doc_or_docs = self.read_single_document(
                file_path,
                chunk_size=chunk_size,
                overlap=overlap,
                token_based_chunking=token_based_chunking,
                
                csv_content_columns=csv_content_columns,
                csv_metadata_columns=csv_metadata_columns,
                csv_delimiter=csv_delimiter,
                csv_encoding=csv_encoding,
                csv_skip_empty_lines=csv_skip_empty_lines
            )
            if doc_or_docs:
                if isinstance(doc_or_docs, list):
                    documents_to_add.extend(doc_or_docs)
                else:
                    documents_to_add.append(doc_or_docs) # Should ideally be list now after chunking all

        print(f"Successfully processed {len(documents_to_add)} individual records/documents from files")

        if documents_to_add:
            # Use batch_size_chroma for the ChromaDB batching
            self.add_documents_to_collection(documents_to_add, collection_name, batch_size=batch_size_chroma)
            print(f"Finished adding documents to collection '{collection_name}'")
        else:
            print("No documents to add after processing directory.")

    def query_documents(self, query: str, collection_name: str = "my_documents", n_results: int = 4, 
                       enable_reranking: bool = True, document_type_filter: str = None):
        """
        Enhanced query system with document classification, filtering, and optional re-ranking.
        
        Args:
            query: The search query
            collection_name: Name of the collection to search
            n_results: Number of results to return
            enable_reranking: Whether to apply re-ranking based on query-document category matching
            document_type_filter: Optional filter to restrict search to specific document types
        """
        try:
            collection = self.client.get_collection(name=collection_name, embedding_function=self.default_embedding_function)
        except Exception as e:
            error_message = f"Collection not found: {collection_name}. Error: {e}"
            print(error_message)
            return [error_message]  # Return error as list for consistency

        # Classify the query to understand user intent
        query_category = self.classifier.classify_query(query)
        print(f"Query classified as: {query_category}")
        
        # Determine document type filter
        if document_type_filter is None and query_category != 'unknown':
            document_type_filter = query_category
        
        # Build where clause for filtering
        where_clause = {}
        if document_type_filter:
            where_clause['document_category'] = document_type_filter
            print(f"Filtering by document category: {document_type_filter}")
        
        # Perform the search with filtering
        search_params = {
            'query_texts': [query],
            'n_results': n_results * 2 if enable_reranking else n_results  # Get more results for re-ranking
        }
        
        if where_clause:
            search_params['where'] = where_clause
        
        try:
            results = collection.query(**search_params)
        except Exception as e:
            print(f"Query failed with filter, trying without filter: {e}")
            # Fallback to unfiltered search
            results = collection.query(
                query_texts=[query],
                n_results=n_results * 2 if enable_reranking else n_results
            )
        
        # Apply re-ranking if enabled
        if enable_reranking and results and results['documents'] and results['documents'][0]:
            results = self._rerank_results(results, query, query_category)
            # Trim to desired number of results
            if len(results['documents'][0]) > n_results:
                for key in results:
                    if isinstance(results[key], list) and len(results[key]) > 0:
                        results[key][0] = results[key][0][:n_results]

        # Format and return results
        return self._format_query_results(results, query_category)
    
    def _rerank_results(self, results: Dict, query: str, query_category: str) -> Dict:
        """
        Re-rank results based on query-document category matching and other factors.
        """
        if not results or not results['documents'] or not results['documents'][0]:
            return results
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results.get('distances', [None])[0] if results.get('distances') else [None] * len(documents)
        
        # Calculate re-ranking scores
        scored_results = []
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            score = self._calculate_relevance_score(doc, metadata, query, query_category, distance)
            scored_results.append({
                'index': i,
                'score': score,
                'document': doc,
                'metadata': metadata,
                'distance': distance
            })
        
        # Sort by score (higher is better)
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Reconstruct results in new order
        reranked_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]] if distances[0] is not None else None
        }
        
        for item in scored_results:
            reranked_results['documents'][0].append(item['document'])
            reranked_results['metadatas'][0].append(item['metadata'])
            if reranked_results['distances']:
                reranked_results['distances'][0].append(item['distance'])
        
        return reranked_results
    
    def _calculate_relevance_score(self, document: str, metadata: Dict, query: str, 
                                  query_category: str, distance: float) -> float:
        """
        Calculate relevance score for re-ranking based on multiple factors.
        """
        score = 0.0
        
        # Base score from embedding similarity (lower distance = higher score)
        if distance is not None:
            score += max(0, 1.0 - distance)
        else:
            score += 0.5  # Default if no distance available
        
        # Category matching bonus
        doc_category = metadata.get('document_category', 'default')
        if doc_category == query_category:
            score += 0.3
        
        # Tag matching bonus
        doc_tags = metadata.get('document_tags', [])
        query_lower = query.lower()
        for tag in doc_tags:
            if tag.lower() in query_lower:
                score += 0.1
        
        # Content length preference (moderate length preferred)
        content_length = len(document)
        if 200 <= content_length <= 1500:
            score += 0.1
        elif content_length < 100:
            score -= 0.1
        
        # Specific category bonuses
        if query_category == 'literature':
            if metadata.get('has_dialogue', False):
                score += 0.05
            if 'character' in query_lower and 'character_analysis' in doc_tags:
                score += 0.2
        
        elif query_category == 'economics':
            if metadata.get('has_percentages', False) and any(char in query for char in ['%', 'percent']):
                score += 0.1
            if metadata.get('numeric_density', 0) > 0.05:  # High numeric content
                score += 0.1
        
        elif query_category == 'technical':
            if metadata.get('has_code', False) and any(word in query_lower for word in ['code', 'function', 'method']):
                score += 0.1
            if 'documentation' in doc_tags and 'documentation' in query_lower:
                score += 0.2
        
        return score
    
    def _format_query_results(self, results: Dict, query_category: str) -> List[str]:
        """
        Format query results for display with enhanced information.
        """
        output_lines = [f"\nTop Results (Query Category: {query_category}):"]
        
        if results and results['documents'] and results['documents'][0]:
            for i, doc_content in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                output_lines.append(f"\nResult {i + 1}:")
                output_lines.append(f"  File: {metadata.get('file_name', 'N/A')}")
                output_lines.append(f"  File Type: {metadata.get('file_type', 'N/A')}")
                output_lines.append(f"  Document Category: {metadata.get('document_category', 'N/A')}")
                output_lines.append(f"  Document Tags: {', '.join(metadata.get('document_tags', []))}")
                output_lines.append(f"  Chunk Index: {metadata.get('chunk_index', 'N/A')}")
                output_lines.append(f"  Content (Preview): {doc_content}")
                
                # Special handling for CSV original row data, if present
                if metadata.get('file_type') == 'csv' and 'original_row_data' in metadata:
                    try:
                        # original_row_data would have been stored as JSON string
                        original_row_data = json.loads(metadata['original_row_data'])
                        output_lines.append("  Original CSV Row Data:")
                        for k, v in original_row_data.items():
                            output_lines.append(f"    - {k}: {v}")
                    except json.JSONDecodeError:
                        output_lines.append(f"  Original CSV Row Data (Error parsing JSON): {metadata['original_row_data']}")
                
                # Enhanced metadata display
                output_lines.append("  Enhanced Metadata:")
                enhanced_keys = ['content_hash', 'has_numbers', 'has_currency', 'sentence_count', 
                               'has_dialogue', 'proper_nouns', 'has_percentages', 'has_dates', 
                               'numeric_density', 'has_code', 'has_urls', 'technical_terms']
                
                for key in enhanced_keys:
                    if key in metadata:
                        output_lines.append(f"    - {key}: {metadata[key]}")
                
                # Generic metadata print for other cases
                output_lines.append("  Other Metadata:")
                skip_keys = ['file_name', 'file_type', 'chunk_index', 'original_row_data', 
                           'document_category', 'document_tags'] + enhanced_keys
                for k, v in metadata.items():
                    if k not in skip_keys:
                        output_lines.append(f"    - {k}: {v}")
        else:
            output_lines.append("\nNo results found for your query.")
            if query_category != 'unknown':
                output_lines.append(f"Try searching without category filter or check if you have {query_category} documents in your collection.")
        
        return output_lines

    def query_documents_by_category(self, query: str, category: str, collection_name: str = "my_documents", 
                                   n_results: int = 4):
        """
        Query documents filtered by a specific category.
        """
        return self.query_documents(
            query=query,
            collection_name=collection_name,
            n_results=n_results,
            enable_reranking=True,
            document_type_filter=category
        )
    
    def get_document_categories(self, collection_name: str = "my_documents") -> Dict[str, int]:
        """
        Get statistics about document categories in the collection.
        """
        try:
            collection = self.client.get_collection(name=collection_name, embedding_function=self.default_embedding_function)
            
            # Get all documents to analyze categories
            all_results = collection.get()
            
            category_counts = {}
            if all_results and all_results['metadatas']:
                for metadata in all_results['metadatas']:
                    category = metadata.get('document_category', 'unknown')
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            return category_counts
        except Exception as e:
            print(f"Error getting document categories: {e}")
            return {}
    
    def get_document_content(self, query: str, collection_name: str = "my_documents", n_results: int = 4, 
                            enable_reranking: bool = True, document_type_filter: str = None) -> List[str]:
        """
        Get just the document content without formatting - for use in chat systems.
        
        Args:
            query: The search query
            collection_name: Name of the collection to search
            n_results: Number of results to return
            enable_reranking: Whether to apply re-ranking based on query-document category matching
            document_type_filter: Optional filter to restrict search to specific document types
            
        Returns:
            List of document content strings
        """
        try:
            collection = self.client.get_collection(name=collection_name, embedding_function=self.default_embedding_function)
        except Exception as e:
            error_message = f"Collection not found: {collection_name}. Error: {e}"
            print(error_message)
            return [error_message]  # Return error as list for consistency

        # Classify the query to understand user intent
        query_category = self.classifier.classify_query(query)
        print(f"Query classified as: {query_category}")
        
        # Determine document type filter
        if document_type_filter is None and query_category != 'unknown':
            document_type_filter = query_category
        
        # Build where clause for filtering
        where_clause = {}
        if document_type_filter:
            where_clause['document_category'] = document_type_filter
            print(f"Filtering by document category: {document_type_filter}")
        
        # Perform the search with filtering
        search_params = {
            'query_texts': [query],
            'n_results': n_results * 2 if enable_reranking else n_results  # Get more results for re-ranking
        }
        
        if where_clause:
            search_params['where'] = where_clause
        
        try:
            results = collection.query(**search_params)
        except Exception as e:
            print(f"Query failed with filter, trying without filter: {e}")
            # Fallback to unfiltered search
            results = collection.query(
                query_texts=[query],
                n_results=n_results * 2 if enable_reranking else n_results
            )
        
        # Apply re-ranking if enabled
        if enable_reranking and results and results['documents'] and results['documents'][0]:
            results = self._rerank_results(results, query, query_category)
            # Trim to desired number of results
            if len(results['documents'][0]) > n_results:
                for key in results:
                    if isinstance(results[key], list) and len(results[key]) > 0:
                        results[key][0] = results[key][0][:n_results]

        # Return just the document content
        if results and results['documents'] and results['documents'][0]:
            return results['documents'][0]
        else:
            return [f"No results found for query: {query}"]


# VectorStoreIndex + ChromaDB integration code example:
    # from llama_index.vector_stores.chroma import ChromaVectorStore
    # from llama_index import VectorStoreIndex
    # from chromadb import PersistentClient
    # chroma_client = PersistentClient(path="./chroma_db")
    # vector_store = ChromaVectorStore(chroma_client=chroma_client, collection_name="my_collection")
    # index = VectorStoreIndex.from_vector_store(vector_store)

# SimpleDirectoryReader: reads documents from a directory and processes them into chunks (like scan_directory + read_single_document)

# IngestionPipeline with custom transformations: orchestrator for the document ingestion process

# SimilarityPostprocessor: postprocessing hybrid search, reranking, and postprocessing (e.g., similarity postprocessors, response synthesizers)