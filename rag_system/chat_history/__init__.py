"""
Chat History Package

Contains chat history management components for different storage backends.
"""

# Import chat history managers with error handling
try:
    from .chroma_chat_history import ChatHistoryManager as ChromaChatHistoryManager
except ImportError as e:
    print(f"⚠️  Could not import ChromaChatHistoryManager: {e}")
    ChromaChatHistoryManager = None


try:
    from .mongo_chat_history import MongoDBChatHistoryManager
except ImportError as e:
    print(f"⚠️  Could not import MongoDBChatHistoryManager: {e}")
    MongoDBChatHistoryManager = None


__all__ = [
    'ChromaChatHistoryManager',
    'MongoDBChatHistoryManager',
]

def get_available_chat_backends():
    """Return a list of available chat history backends"""
    available = []
    if ChromaChatHistoryManager:
        available.append('chroma')
    if MongoDBChatHistoryManager:
        available.append('mongodb')
    return available