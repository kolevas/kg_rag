from rag_system.base_engine import BaseChatEngine
class DocumentChatBot(BaseChatEngine):
    """English-only document chat bot using KG-RAG."""

    _CHROMA_DB_PATH = "./rag_system/chroma_db"
    _KG_PATH = "./rag_system/knowledge_graph"
