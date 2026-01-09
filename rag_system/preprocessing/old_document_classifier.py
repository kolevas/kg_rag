"""
Document Classification and Tagging System
Separate module for document classification logic
"""
import re
from typing import Dict, List, Any
from pathlib import Path


class DocumentClassifier:
    def __init__(self):
        # Simplified classification patterns
        self.patterns = {
            'literature': ['novel', 'character', 'pride', 'prejudice', 'austen', 'story', 'fiction', 'elizabeth', 'bennet'],
            'economics': ['gdp', 'economic', 'finance', 'market', 'inflation', 'currency', 'trade'],
            'technical': ['readme', 'code', 'programming', 'software', 'api', 'function', 'development', 'install'],
            'presentation': ['slide', 'presentation', 'ppt', 'overview', 'summary']
        }

    def classify_document(self, file_path: str, content: str = None) -> str:
        """Classify document based on file path and content."""
        text = f"{file_path} {content or ''}".lower()
        
        for category, keywords in self.patterns.items():
            if any(keyword in text for keyword in keywords):
                return category
        return 'default'

    def classify_query(self, query: str) -> str:
        """Classify user query to determine document type."""
        query_lower = query.lower()
        
        for category, keywords in self.patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return 'unknown'

    def enhance_metadata(self, metadata: Dict[str, Any], document_category: str, content: str) -> Dict[str, Any]:
        """Enhance metadata with classification and basic content analysis."""
        enhanced = metadata.copy()
        enhanced['document_category'] = document_category
        enhanced['content_length'] = len(content)
        enhanced['word_count'] = len(content.split())
        enhanced['has_numbers'] = bool(re.search(r'\d+', content))
        
        return enhanced
