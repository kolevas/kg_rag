"""
Preprocessing Package

Contains document preprocessing and ingestion components.
"""

# Import preprocessing components with error handling
try:
    from .document_reader import DocumentReader
except ImportError as e:
    print(f"⚠️  Could not import DocumentReader: {e}")
    DocumentReader = None

from .text_utils import (
    clean_text,
    split_text_into_chunks,
    normalize_chunk,
)


__all__ = [
    'DocumentReader',
    'clean_text',
    'split_text_into_chunks',
    'normalize_chunk',
]