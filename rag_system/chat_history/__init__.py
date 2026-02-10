"""
Chat History Package

Contains chat history management components for different storage backends.
"""

try:
    from .mongo_chat_history import MongoDBChatHistoryManager
except ImportError as e:
    print(f"⚠️  Could not import MongoDBChatHistoryManager: {e}")
    MongoDBChatHistoryManager = None


__all__ = [
    'MongoDBChatHistoryManager',
]

def get_available_chat_backends():
    """Return a list of available chat history backends"""
    available = []
    if MongoDBChatHistoryManager:
        available.append('mongodb')
    return available