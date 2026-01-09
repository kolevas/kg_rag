import pdfplumber
import re
from typing import List, Dict, Optional, Tuple, Any
import unicodedata
from pathlib import Path
import time
import fitz  # PyMuPDF
import os
import io

import pdfplumber.page

def extract_ordered_content_to_text(file) -> str:
    """
    Extracts and orders content (text and tables) from a PDF file
    and returns it as a single coherent string.
    """
    all_ordered_elements_on_all_pages = []
    print(f"Type: {type(file['content'])}, Length: {len(file['content'])}")
    print(f"First 10 bytes: {file['content'][:10]}")


    try:
        with pdfplumber.open(io.BytesIO(file["content"])) as pdf_plumber:
            with fitz.open(stream=file['content'], filetype="pdf") as pdf_fitz:
                # total_pages = len(pdf_plumber.pages)

                for page_num, plumber_page in enumerate(pdf_plumber.pages):
                    #logging.info(f"Processing page {page_num + 1}/{total_pages}...")
                    elements_on_current_page = []
                    fitz_page = pdf_fitz[page_num]

                    # --- 1. Extract tables using pdfplumber ---
                    tables = plumber_page.find_tables()
                    table_bboxes = [(t.bbox, t) for t in tables] 

                    for bbox, table_obj in table_bboxes:
                        try:
                            table_data = table_obj.extract() 
                            # Only process non-empty tables
                            if not table_data or not any(any(cell for cell in row) for row in table_data):
                                continue  # Skip empty tables
                            markdown_table = ""
                            if table_data:
                                # Format table data into Markdown
                                headers = table_data[0] if table_data and len(table_data) > 0 else []
                                rows = table_data[1:] if table_data and len(table_data) > 1 else []
                                
                                if headers:
                                    markdown_table += "| " + " | ".join(str(h).strip() if h is not None else "" for h in headers) + " |\n"
                                    markdown_table += "|---" * len(headers) + "|\n"
                                
                                for row in rows:
                                    markdown_table += "| " + " | ".join(str(c).strip() if c is not None else "" for c in row) + " |\n"
                            
                            if markdown_table.strip(): # Only add if there's actual content
                                
                                elements_on_current_page.append({
                                    'type': 'table',
                                    'content': f"\nTable Content:\n{markdown_table.strip()}\nEnd of table content.\n\n", # Add marker and content
                                    'bbox': bbox,
                                    'page_num': page_num + 1,
                                    'order_key': bbox[1] # Use y0 for initial vertical sorting
                                })
                        except Exception as e:
                            print(f"  Error extracting table on Page {page_num + 1} with bbox {bbox}: {e}")


                    # --- 2. Extract and process text into paragraphs using PyMuPDF ---
                    all_fitz_text_elements = []
                    try:
                        fitz_words = fitz_page.get_text("words")
                        fitz_words = [w for w in fitz_words if len(w) >= 5] # Ensure word object has all expected parts
                        
                        for word_obj in fitz_words:
                            x0, y0, x1, y1, text_content = word_obj[0], word_obj[1], word_obj[2], word_obj[3], word_obj[4]
                            all_fitz_text_elements.append({
                                'text': text_content,
                                'x0': x0, 'y0': y0, 
                                'x1': x1, 'y1': y1,
                                'height': y1 - y0 # Needed for paragraph threshold
                            })
                    except Exception as e:
                        print(f"  Error extracting words/text elements on Page {page_num + 1} with PyMuPDF: {e}")

                    current_paragraph_words = [] 
                    
                    if all_fitz_text_elements:
                        all_fitz_text_elements.sort(key=lambda x: (x['y0'], x['x0']))

                        # Estimate average character height for dynamic paragraph breaking
                        # Use a default if no words or words have zero height
                        avg_char_height = sum(w['height'] for w in all_fitz_text_elements) / len(all_fitz_text_elements) if all_fitz_text_elements and sum(w['height'] for w in all_fitz_text_elements) > 0 else 10
                        PARA_BREAK_THRESHOLD = avg_char_height * 1.5

                        for text_word_obj in all_fitz_text_elements:
                            is_part_of_table = False
                            for table_bbox, _ in table_bboxes:
                                # Check for overlap. A word is considered part of a table if its bounding box
                                # significantly overlaps with a table's bounding box.
                                if (text_word_obj['x0'] < table_bbox[2] and text_word_obj['x1'] > table_bbox[0] and
                                    text_word_obj['y0'] < table_bbox[3] and text_word_obj['y1'] > table_bbox[1]):
                                    is_part_of_table = True
                                    break 

                            if is_part_of_table:
                                continue # Skip text that is part of a table

                            if current_paragraph_words:
                                last_word_info = current_paragraph_words[-1]
                                x_tolerance = (text_word_obj['x1'] - text_word_obj['x0']) * 0.5 # Half word width

                                if (text_word_obj['y0'] - last_word_info['y1'] > PARA_BREAK_THRESHOLD or
                                    (text_word_obj['y0'] > last_word_info['y0'] + PARA_BREAK_THRESHOLD and 
                                     text_word_obj['x0'] < last_word_info['x0'] - x_tolerance)):
                                    
                                    min_x = min(item['x0'] for item in current_paragraph_words)
                                    min_y = min(item['y0'] for item in current_paragraph_words)
                                    max_x = max(item['x1'] for item in current_paragraph_words)
                                    max_y = max(item['y1'] for item in current_paragraph_words)

                                    elements_on_current_page.append({
                                        'type': 'paragraph',
                                        'content': " ".join([item['text'] for item in current_paragraph_words]).strip(),
                                        'bbox': (min_x, min_y, max_x, max_y),
                                        'page_num': page_num + 1,
                                        'order_key': min_y
                                    })
                                    current_paragraph_words = []
                                
                            current_paragraph_words.append(text_word_obj)

                        # Add any remaining words as the last paragraph on the page
                        if current_paragraph_words:
                             min_x = min(item['x0'] for item in current_paragraph_words)
                             min_y = min(item['y0'] for item in current_paragraph_words)
                             max_x = max(item['x1'] for item in current_paragraph_words)
                             max_y = max(item['y1'] for item in current_paragraph_words)

                             elements_on_current_page.append({
                                'type': 'paragraph',
                                'content': " ".join([item['text'] for item in current_paragraph_words]).strip(),
                                'bbox': (min_x, min_y, max_x, max_y),
                                'page_num': page_num + 1,
                                'order_key': min_y
                            })
                    else: 
                        print(f"  No non-table text elements found for paragraph processing on Page {page_num + 1}.")

                    # Sort elements on the current page by their vertical position
                    elements_on_current_page.sort(key=lambda x: x['order_key']) 
                    all_ordered_elements_on_all_pages.extend(elements_on_current_page)
                    
    except FileNotFoundError:
        print(f"The file '{file['name']}' was not found.")
        # return ""
    except Exception as e:
        print(f"An unexpected error occurred during PDF processing for {file['name']}")
        print(e)
        # return ""

    full_text_content = ""
    for element in all_ordered_elements_on_all_pages:
        # Tables already have their marker, paragraphs need cleaning and then concatenation
        if element['type'] == 'paragraph':
            full_text_content += clean_text(element['content']) + "\n\n"
        elif element['type'] == 'table':
            full_text_content += element['content'] + "\n\n" # Table content already formatted with marker
            
    return full_text_content.strip()

def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing extra whitespace, special characters,
    and normalizing unicode characters. Preserves table formatting.
    """
    parts = text.split("\nTable Content:")
    cleaned_parts = []
    if parts[0].strip():
        cleaned = unicodedata.normalize('NFKD', parts[0])
        cleaned = re.sub(r'[^\w\s.,!?-]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned_parts.append(cleaned.strip())
    
    for part in parts[1:]:
        if part.strip():
            cleaned_parts.append("\nTable Content:" + part)
    
    return '\n'.join(cleaned_parts)

def split_text_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks of specified size.
    - If a table starts in a chunk and cannot be stored as a whole, move it to the next chunk.
    - If a table is larger than the chunk size, split by rows, each chunk starts with the header.
    - If a table fits in the chunk, keep it with other data in the same chunk.
    """
    chunks = []
    text_length = len(text)
    start = 0

    min_overlap = max(50, int(overlap * 0.5)) # Ensure a minimum overlap, at least 50
    max_overlap = min(overlap * 1.5, chunk_size - 50) # Max overlap should not be too close to chunk_size

    # Find all tables
    table_matches = list(re.finditer(r"(Table Content:.*?End of table content\.)", text, re.DOTALL))
    table_spans = [m.span() for m in table_matches]

    while start < text_length:
        # Find the next table that starts after 'start'
        next_table = None
        for t_start, t_end in table_spans:
            if t_start >= start:
                next_table = (t_start, t_end)
                break

        end = start + chunk_size
        if end > text_length:
            end = text_length

        if next_table and next_table[0] < end:
            t_start, t_end = next_table
            # If the table fits entirely within the chunk, include it with other data
            if t_end <= end:
                # Find proper sentence boundary before the table or at the end
                best_end = end
                if end < text_length:
                    for i in range(end, max(start + chunk_size // 2, start), -1):
                        if i < text_length and text[i-1] in '.!?' and (i == text_length or text[i].isspace() or text[i].isupper()):
                            best_end = i
                            break
                
                chunk = text[start:best_end]
                chunks.append(chunk)
                print(f"Created chunk {len(chunks)} (length: {len(chunk)})")
                if best_end >= text_length:
                    break
                
                # Find where to start the next chunk (after any whitespace)
                next_start = best_end
                while next_start < text_length and text[next_start].isspace():
                    next_start += 1
                
                start = next_start
            else:
                # Table would be split, so move it to the next chunk
                if t_start > start:
                    # Add text before the table as a chunk (use original logic)
                    pre_table_text = text[start:t_start]
                    if pre_table_text.strip():
                        pre_chunks = _original_chunking(pre_table_text, chunk_size, overlap)
                        chunks.extend(pre_chunks)
                # Now handle the table
                if t_end - t_start > chunk_size:
                    # Table is too big, split by rows with header
                    table_text = text[t_start:t_end]
                    table_lines = table_text.split('\n')
                    header = table_lines[1].strip() if len(table_lines) > 1 else ''
                    rows = table_lines[2:-1]
                    current_chunk = f"Table Content:\n{header}"
                    for row in rows:
                        if len(current_chunk) + len(row) + 1 > chunk_size:
                            chunks.append(current_chunk)
                            print(f"Created chunk {len(chunks)} (length: {len(current_chunk)})")
                            current_chunk = f"Table Content:\n{header}\n{row}"
                        else:
                            current_chunk += f"\n{row}"
                    if current_chunk:
                        chunks.append(current_chunk + "\nEnd of table content.")
                        print(f"Created chunk {len(chunks)} (length: {len(current_chunk)})")
                else:
                    # Table fits as a whole, add it as a chunk
                    chunks.append(text[t_start:t_end])
                    print(f"Created chunk {len(chunks)} (length: {len(text[t_start:t_end])})")
                start = t_end
        else:
            # No table in this chunk, find proper sentence boundaries
            end = start + chunk_size
            if end >= text_length:
                end = text_length
                chunk = text[start:end]
                chunks.append(chunk)
                break
            else:
                # Find the best place to end the chunk at a sentence boundary
                best_end = end
                # Look backwards from the target end to find a sentence ending
                for i in range(end, max(start + chunk_size // 2, start), -1):
                    if i < text_length and text[i-1] in '.!?' and (i == text_length or text[i].isspace() or text[i].isupper()):
                        best_end = i
                        break
                
                # If no sentence boundary found in reasonable range, look for other boundaries
                if best_end == end:
                    for i in range(end, max(start + chunk_size // 2, start), -1):
                        if i < text_length and text[i-1] in '.\n':
                            best_end = i
                            break
                
                chunk = text[start:best_end]
                chunks.append(chunk)
                print(f"Created chunk {len(chunks)} (length: {len(chunk)})")
                
                # Find where to start the next chunk (after any whitespace)
                next_start = best_end
                while next_start < text_length and text[next_start].isspace():
                    next_start += 1
                
                start = next_start

    print(f"Text splitting completed. Total chunks: {len(chunks)}")
    return chunks

def _original_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    text_length = len(text)
    start = 0
    min_overlap = max(50, int(overlap * 0.5)) # Ensure a minimum overlap, at least 50
    max_overlap = min(overlap * 1.5, chunk_size - 50) # Max overlap should not be too close to chunk_size
    while start < text_length:
        end = start + chunk_size
        if end < text_length:
            # Look for sentence boundaries first (period, exclamation, question mark)
            sentence_end = -1
            for i in range(end, start, -1):
                if text[i-1] in '.!?' and i < len(text) and (text[i].isspace() or text[i].isupper()):
                    sentence_end = i
                    break
            
            # If no sentence boundary found, fall back to other boundaries
            if sentence_end == -1:
                last_period = text.rfind('.', start, end)
                last_exclamation = text.rfind('!', start, end)
                last_question = text.rfind('?', start, end)
                last_newline = text.rfind('\n', start, end)
                last_space = text.rfind(' ', start, end)
                split_point = max(last_period, last_exclamation, last_question, last_newline, last_space)
            else:
                split_point = sentence_end
            if split_point > start:
                end = split_point + 1
        else:
            end = text_length
        chunk = text[start:end]
        chunks.append(chunk)
        print(f"Created chunk {len(chunks)} (length: {len(chunk)})")
        if end == text_length:
            break
        potential_overlap_start = max(start, end - overlap)
        overlap_candidate = -1
        for i in range(end - min_overlap, potential_overlap_start -1, -1):
            if text[i] in '.!?' and i + 1 < len(text) and (text[i + 1].isspace() or text[i + 1].isupper()):
                overlap_candidate = i + 1  # Start after the sentence end
                break
            elif text[i] == '.' or text[i] == ' ' or text[i] == '\n':
                overlap_candidate = i + 1
                break
        current_overlap = overlap
        if overlap_candidate != -1:
            calculated_overlap = end - overlap_candidate
            if min_overlap <= calculated_overlap <= max_overlap:
                current_overlap = calculated_overlap
            elif calculated_overlap < min_overlap:
                current_overlap = min_overlap
            else:
                current_overlap = max_overlap
        else:
            current_overlap = max(min_overlap, min(overlap, max_overlap))
            print(f"No strategic overlap point found. Using default/clamped overlap: {current_overlap}")
        start = end - current_overlap
    return chunks

def normalize_chunk(chunk: str) -> str:
    normalized_response = re.sub(r"\[\[\d+\]\],?\s*", "", chunk)
    normalized_response = re.sub(r"\n\s*#{1,6}\s*([^\n]+)", r". \1:", normalized_response)
    normalized_response = re.sub(r"\n\s*-\s*\*\*([^*]+)\*\*:\s*", r". \1: ", normalized_response)
    normalized_response = re.sub(r"\n\s*-\s*", ". ", normalized_response)
    normalized_response = re.sub(r"\*\*([^*]+)\*\*", r"\1", normalized_response)
    normalized_response = re.sub(r"\*([^*]+)\*", r"\1", normalized_response)
    normalized_response = re.sub(r"\n{3,}", "\n\n", normalized_response)
    normalized_response = re.sub(r"\n\n", ". ", normalized_response)
    normalized_response = re.sub(r"\n", " ", normalized_response)
    # NEW: Handle hyphenated words broken across lines
    normalized_response = re.sub(r"(\w)-\s+(\w)", r"\1\2", normalized_response)
    
    # NEW: Remove extra hyphens and normalize dashes
    normalized_response = re.sub(r"--+", "—", normalized_response)
    
    # NEW: Fix common PDF extraction artifacts
    normalized_response = re.sub(r"\s*\|\s*", " ", normalized_response)  # Remove table separators
    normalized_response = re.sub(r"(\w)\s*_\s*(\w)", r"\1_\2", normalized_response)  # Fix broken underscores
    
    # NEW: Handle page numbers and headers/footers patterns
    normalized_response = re.sub(r"\b(Page\s+\d+|©\s*\d+|All rights reserved)\b", "", normalized_response, flags=re.IGNORECASE)
    
    # NEW: Fix spacing around parentheses and brackets
    normalized_response = re.sub(r"\s*\(\s*", " (", normalized_response)
    normalized_response = re.sub(r"\s*\)\s*", ") ", normalized_response)
    normalized_response = re.sub(r"\s*\[\s*", " [", normalized_response)
    normalized_response = re.sub(r"\s*\]\s*", "] ", normalized_response)
    
    # ...existing code...
    normalized_response = re.sub(r"\.+", ".", normalized_response)
    normalized_response = re.sub(r"\s*\.\s*\.", ".", normalized_response)
    normalized_response = re.sub(r":\s*\.", ":", normalized_response)
    normalized_response = re.sub(r"\.\s*:", ":", normalized_response)
    
    # NEW: Clean up multiple punctuation marks
    normalized_response = re.sub(r"[.]{2,}", ".", normalized_response)
    normalized_response = re.sub(r"[,]{2,}", ",", normalized_response)
    
    normalized_response = re.sub(r"\s+", " ", normalized_response)
    normalized_response = re.sub(r"\s*([.,:;!?])", r"\1", normalized_response)
    normalized_response = re.sub(r"([.,:;!?])\s*", r"\1 ", normalized_response)
    
    # NEW: Fix spacing before opening quotes and after closing quotes
    normalized_response = re.sub(r'\s*"([^"]*)"\s*', r' "\1" ', normalized_response)
    normalized_response = re.sub(r"\s*'([^']*)'\s*", r" '\1' ", normalized_response)
    
    normalized_response = re.sub(r"\.\s*([a-z])", lambda m: ". " + m.group(1).upper(), normalized_response)
    normalized_response = re.sub(r"(\d+)\.\s+(\d+)", r"\1.\2", normalized_response)
    
    # NEW: Final cleanup - remove any remaining double spaces
    normalized_response = re.sub(r"\s{2,}", " ", normalized_response)
    
    normalized_response = normalized_response.strip()
    if normalized_response and not normalized_response[0].isupper():
        normalized_response = normalized_response[0].upper() + normalized_response[1:]
    if normalized_response and normalized_response[-1] not in '.!?':
        normalized_response += "."
    return normalized_response

def preprocess_pdf(file, chunk_size: int = 1500, overlap: int = 200, blob_metadata=None) -> Dict[str, List[str]]:
    """
    Main preprocessing function that handles the entire PDF preprocessing pipeline.
    """
    print(f"\nStarting PDF preprocessing: {file["name"]}")
    
    print("\nStep 1: Extracting text from PDF")
    content_text = extract_ordered_content_to_text(file)
    print(f"Content size = {len(content_text)}")
    
    print("\nStep 2: Splitting into chunks")
    chunks = split_text_into_chunks(content_text, chunk_size, overlap)
    for chunk in chunks:
        chunk = normalize_chunk(chunk)
        print(f"Normalized chunk: {chunk}")
    print(f"Chunk length for file {file['name']} = {len(chunks)}")
    
    # Prepare metadata
    metadata = {
        'user_id': 'example_user',
        'client_id': 'example_client',
        'timestamp_in_seconds': time.time(),
        'num_chunks': len(chunks),
        'total_chars': len(content_text)
    }
    metadata.update(blob_metadata or {}) 
    metadata["has_been_preprocessed"] = "True" # Merge with any provided metadata
    
    print("\nPreprocessing completed successfully!")
    print("Preprocessed document chunks:")
    return {
        'result':{
            'chunks': chunks,
            'metadata': metadata
        }
        
    }

if __name__ == "__main__":
    # Example usage
    try:
        doc_path = r"/Users/snezhanakoleva/praksa/local_document_preprocessing/test_data/Onculitis Real-World Evidence and Long-Term Safety Monitoring Advisory Board.pdf"
        print("\n=== Starting PDF Processing ===")
        start_time = time.time()
        
        # Read the file and create the expected dictionary format
        with open(doc_path, 'rb') as f:
            file_content = f.read()
        
        file_dict = {"name": doc_path, "content": file_content}
        result = preprocess_pdf(file_dict)

        print("\n=== Processing Results ===")
        print(f"Processed document: {result['result']['metadata']['user_id']}")
        print(f"Number of chunks: {result['result']['metadata']['num_chunks']}")
        print(f"Total characters: {result['result']['metadata']['total_chars']}")
        print(f"Processing time: {time.time() - start_time:.2f} seconds")
        # Print first chunk as example
        # if result['result']['chunks']:
        #     for chunk in result['result']['chunks']:
        #         print(f"\nChunk:\n{chunk}\n")

            
    except Exception as e:
        print(f"\nError processing PDF: {str(e)}")
