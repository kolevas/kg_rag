import re
import unicodedata
from typing import List


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
   
    chunks = []
    text_length = len(text)
    start = 0

    while start < text_length:
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
            sentence_end = -1
            for i in range(end, max(start + chunk_size // 2, start), -1):
                if i < text_length and text[i-1] in '.!?' and (i == text_length or text[i].isspace() or text[i].isupper()):
                    sentence_end = i
                    break
            
            if sentence_end != -1:
                best_end = sentence_end
            else:
                # If no sentence boundary found, look for other boundaries
                for i in range(end, max(start + chunk_size // 2, start), -1):
                    if i < text_length and text[i-1] in '.\n':
                        best_end = i
                        break
            
            chunk = text[start:best_end]
            chunks.append(chunk)
            
            if best_end >= text_length:
                break
            
            # Find where to start the next chunk (after any whitespace)
            next_start = best_end
            while next_start < text_length and text[next_start].isspace():
                next_start += 1
            
            start = next_start
    
    return chunks


_MARKDOWN_RULES = [
    (r"\[\[\d+\]\],?\s*", ""),               
    (r"\n\s*#{1,6}\s*([^\n]+)", r". \1:"),           
    (r"\n\s*-\s*\*\*([^*]+)\*\*:\s*", r". \1: "),    
    (r"\n\s*-\s*", ". "),                              
    (r"\*\*([^*]+)\*\*", r"\1"),                      
    (r"\*([^*]+)\*", r"\1"),                           
]

_WHITESPACE_RULES = [
    (r"\n{3,}", "\n\n"),
    (r"\n\n", ". "),
    (r"\n", " "),
    (r"(\w)-\s+(\w)", r"\1\2"),     
    (r"--+", "—"),                     
]

_ARTIFACT_RULES = [
    (r"\s*\|\s*", " "),                                  
    (r"(\w)\s*_\s*(\w)", r"\1_\2"),                          
    (r"\b(Page\s+\d+|©\s*\d+|All rights reserved)\b", ""),   
]

_SPACING_RULES = [
    (r"\s*\(\s*", " ("),
    (r"\s*\)\s*", ") "),
    (r"\s*\[\s*", " ["),
    (r"\s*\]\s*", "] "),
    (r"\s+", " "),
    (r"\s*([.,:;!?])", r"\1"),
    (r"([.,:;!?])\s*", r"\1 "),
    (r'\s*"([^"]*)"\s*', r' "\1" '),
    (r"\s*'([^']*)'\s*", r" '\1' "),
    (r"(\d+)\.\s+(\d+)", r"\1.\2"),  
]


def _apply_rules(text: str, rules: list) -> str:
    """Apply a sequence of (pattern, replacement) regex rules."""
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    return text


def normalize_chunk(chunk: str) -> str:
    """Normalize a text chunk by cleaning up formatting, spacing, and punctuation."""
    text = _apply_rules(chunk, _MARKDOWN_RULES)
    text = _apply_rules(text, _WHITESPACE_RULES)
    text = _apply_rules(text, _ARTIFACT_RULES)
    text = _apply_rules(text, _SPACING_RULES)

    # Capitalize after periods
    text = re.sub(r"\.\s*([a-z])", lambda m: ". " + m.group(1).upper(), text)

    # Final cleanup
    text = re.sub(r"\s{2,}", " ", text).strip()
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?':
        text += "."
    return text