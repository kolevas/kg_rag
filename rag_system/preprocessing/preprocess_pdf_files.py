import pdfplumber
import re
from typing import List, Dict
import time
import fitz  # PyMuPDF
import io
from .text_utils import clean_text, split_text_into_chunks, normalize_chunk

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

# clean_text, split_text_into_chunks, and normalize_chunk
# are now imported from text_utils



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
