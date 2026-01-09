"""
RAG System Package

This package contains components for Retrieval-Augmented Generation including:
- LlamaIndex integration
- Document preprocessing
- Chat history management
- Vector storage with ChromaDB
"""

try:
    from .token_utils import TokenUtils
except ImportError as e:
    print(f"⚠️  Could not import TokenUtils: {e}")
    TokenUtils = None

try:
    from .vanilla_engine import DocumentChatBot as VanillaEngine
except ImportError as e:
    print(f"⚠️  Could not import VanillaEngine: {e}")
    VanillaEngine = None

# Export available components
__all__ = [
    'LlamaIndexEngine',
    'LlamaIndexPreprocessor', 
    'TokenUtils',
    'VanillaEngine'
]

def get_available_rag_components():
    """Return a list of available RAG components"""
    available = []
    if TokenUtils:
        available.append('TokenUtils')
    if VanillaEngine:
        available.append('VanillaEngine')
    return available

def check_rag_dependencies():
    """Check if required dependencies are available"""
    dependencies = {
        'chromadb': False,
        'llama_index': False,
        'openai': False,
        'sentence_transformers': False
    }
    
    try:
        import chromadb
        dependencies['chromadb'] = True
    except ImportError:
        pass
    
    try:
        import llama_index
        dependencies['llama_index'] = True
    except ImportError:
        pass
    
    try:
        import openai
        dependencies['openai'] = True
    except ImportError:
        pass
    
    try:
        import sentence_transformers
        dependencies['sentence_transformers'] = True
    except ImportError:
        pass
    
    return dependencies