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

try:
    from .old_document_classifier import DocumentClassifier
except ImportError as e:
    print(f"⚠️  Could not import DocumentClassifier: {e}")
    DocumentClassifier = None


__all__ = [
    'DocumentReader',
    'DocumentClassifier',
    'DocumentIngester'
]