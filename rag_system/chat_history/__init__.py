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
    from .cosmos_chat_history import CosmosDBChatHistoryManager as CosmosChatHistoryManager
except ImportError as e:
    print(f"⚠️  Could not import CosmosChatHistoryManager: {e}")
    CosmosChatHistoryManager = None

try:
    from .mongo_chat_history import MongoDBChatHistoryManager
except ImportError as e:
    print(f"⚠️  Could not import MongoDBChatHistoryManager: {e}")
    MongoDBChatHistoryManager = None

try:
    from .postgres_chat_history import PostgreSQLChatHistoryManager as PostgresChatHistoryManager
except ImportError as e:
    print(f"⚠️  Could not import PostgresChatHistoryManager: {e}")
    PostgresChatHistoryManager = None

__all__ = [
    'ChromaChatHistoryManager',
    'CosmosChatHistoryManager', 
    'MongoDBChatHistoryManager',
    'PostgresChatHistoryManager'
]

def get_available_chat_backends():
    """Return a list of available chat history backends"""
    available = []
    if ChromaChatHistoryManager:
        available.append('chroma')
    if CosmosChatHistoryManager:
        available.append('cosmos')
    if MongoDBChatHistoryManager:
        available.append('mongodb')
    if PostgresChatHistoryManager:
        available.append('postgres')
    return available