import csv
from typing import List, Dict, Any, Union, Optional, Callable

import os

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text) 
    text = text.lower().strip()
    text = ' '.join(text.split()) 
    return text

def normalize_number(value: Any) -> Optional[Union[int, float]]:
   
    if value is None or value == '':
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return None 

def normalize_boolean(value: Any) -> Optional[bool]:
    if value is None or value == '':
        return None
    lower_value = str(value).lower().strip()
    if lower_value in ['true', '1', 'yes', 'y']:
        return True
    elif lower_value in ['false', '0', 'no', 'n']:
        return False
    return None

def read_csv_file(
    file_path: str,
    content_columns: Optional[List[str]] = None,
    metadata_columns: Optional[List[str]] = None,
    delimiter: str = ',',
    encoding: str = 'utf-8',
    skip_empty_lines: bool = True,
    normalization_config: Optional[Dict[str, Union[str, Callable[[Any], Any]]]] = None
) -> Union[List[Dict[str, Any]], None]:
    
    documents: List[Dict[str, Any]] = []

    normalization_functions: Dict[str, Callable[[Any], Any]] = {
        'text': normalize_text,
        'number': normalize_number,
        'int': lambda x: normalize_number(x) if isinstance(normalize_number(x), int) else None, 
        'float': lambda x: normalize_number(x) if isinstance(normalize_number(x), float) else None, 
        'boolean': normalize_boolean,
    }

    try:
        base_file_metadata = {
            'file_type': 'csv',
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
        }

        with open(file_path, mode='r', newline='', encoding=encoding) as csvfile:
            if skip_empty_lines:
                csvfile = (line for line in csvfile if line.strip())

          
            print(f"Reading CSV file '{file_path}' with header.")
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                print(f"Warning: CSV file '{file_path}' declared as having a header, but no headers were found or file is empty.")
                return None

            if content_columns:
                invalid_content_cols = [col for col in content_columns if col not in fieldnames]
                if invalid_content_cols:
                    print(f"Error: Specified content_columns not found in header: {invalid_content_cols}")
                    return None
            
            if metadata_columns:
                invalid_metadata_cols = [col for col in metadata_columns if col not in fieldnames]
                if invalid_metadata_cols:
                    print(f"Error: Specified metadata_columns not found in header: {invalid_metadata_cols}")
                    return None
            
            processed_normalization_config: Dict[str, Callable[[Any], Any]] = {}
            if normalization_config:
                for col, norm_type_or_func in normalization_config.items():
                    if isinstance(norm_type_or_func, str):
                        if norm_type_or_func in normalization_functions:
                            processed_normalization_config[col] = normalization_functions[norm_type_or_func]
                        else:
                            print(f"Warning: Unknown normalization type '{norm_type_or_func}' for column '{col}'. Skipping normalization for this column.")
                    elif callable(norm_type_or_func):
                        processed_normalization_config[col] = norm_type_or_func
                    else:
                        print(f"Warning: Invalid normalization configuration for column '{col}'. Must be a string type or a callable. Skipping normalization for this column.")


            for row_index, row_dict in enumerate(reader):
                normalized_row_dict = {}
                for key, value in row_dict.items():
                    if key in processed_normalization_config:
                        normalized_value = processed_normalization_config[key](value)
                        normalized_row_dict[key] = normalized_value
                    else:
                        normalized_row_dict[key] = value
                
                
                content_parts = []
                if content_columns:
                    for col in content_columns:
                        value = normalized_row_dict.get(col, '')
                        if value is not None: 
                            content_parts.append(f"{col}: {str(value)}")
                else:
                    for key, value in normalized_row_dict.items():
                        if value is not None:
                            content_parts.append(f"{key}: {str(value)}")
                    
                
                content = " ".join(content_parts).strip() 

                if not content:
                    print(f"Warning: Empty content generated for row index {row_index} in '{file_path}'. Skipping row.")
                    continue 

                metadata_for_document = {
                    **base_file_metadata,
                    'original_row_index': row_index, 
                    'original_row_data': row_dict 
                }
                
                if metadata_columns:
                    for col in metadata_columns:
                        if col in normalized_row_dict:
                            metadata_for_document[col] = normalized_row_dict[col]
                else:
                    for key, value in normalized_row_dict.items():
                        metadata_for_document[key] = normalized_row_dict[key]

                documents.append({
                    'content': content,
                    'metadata': metadata_for_document
                })
            

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the CSV file: {e}")
        return None

    return documents