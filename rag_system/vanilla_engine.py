"""
Vanilla RAG Engine - Standard RAG without Knowledge Graph
Used for comparison and baseline evaluation against KG-RAG system
"""

from base_engine import BaseChatEngine
from preprocessing.simplified_document_reader import DocumentReader
from chat_history.mongo_chat_history import MongoDBChatHistoryManager
import os
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()


class VanillaChatEngine(BaseChatEngine):
    """
    Standard RAG engine that uses only vector-based document retrieval.
    Excludes all knowledge graph functionality for baseline comparison.
    """

    _CHROMA_DB_PATH = "./chroma_db"
    _KG_PATH = "./rag_system/knowledge_graph"

    def __init__(self, user_id="default_user"):
        """Initialize the vanilla RAG engine without KG retriever."""
        self.user_id = user_id
        self.project_id = "1"
        
        # Initialize only document reader (no KG retriever)
        self.reader = DocumentReader(chroma_db_path=self._CHROMA_DB_PATH)
        self.chat_manager = MongoDBChatHistoryManager(
            db_name="chat_history_db", 
            collection_name="conversations"
        )
        self.session_id = self._initialize_session()
        
        # Explicitly set kg_retriever to None (no knowledge graph)
        self.kg_retriever = None
        
        self.client, self.model_name = self._initialize_llm()
        print("✓ Vanilla RAG Engine initialized (without Knowledge Graph)")

    def process_query(self, query: str) -> str:
        """
        Process a query using only vector-based document retrieval.
        
        Args:
            query: User's question (can be in Macedonian or English)
            
        Returns:
            JSON-formatted response with answer and follow-up questions
        """
        # Get conversation history
        relevant_history = self._get_conversation_history(query)
        
        # Get document context (vector search ONLY)
        context_string = self._get_document_context(query)
        
        # Build history text
        history_text = ""
        if relevant_history:
            history_text = f"\n\nRelevant Context:\n{relevant_history}"

        # Build system message WITHOUT knowledge graph context
        system_message = self._build_system_message(
            context_string, 
            kg_context="",  # No KG context for vanilla RAG
            history_text=history_text
        )
        
        # Generate response
        response = self._generate_response(query, system_message)
        
        # Save conversation
        self._save_conversation(query, response)
        
        return response

    def chat_loop(self):
        """Interactive chat loop for vanilla RAG engine."""
        print("=" * 70)
        print("Vanilla RAG Engine (Document Retrieval Only - No Knowledge Graph)")
        print("=" * 70)
        print("Document Chat Bot initialized!")
        print(f"LLM: {self.model_name}")
        print("Mode: VANILLA RAG (baseline without KG)")
        print("Commands: 'quit', 'exit', 'q'")
        print("-" * 70)
        
        try:
            while True:
                query = input("\nEnter your question (or command): ").strip()
                if not query:
                    continue
                if query.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                try:
                    print(f"\nProcessing: '{query}'")
                    print("[Mode: Vanilla RAG - vector search only]")
                    response = self.process_query(query)
                    print("\nResponse:")
                    print(response)
                    print("-" * 70)
                except Exception as e:
                    print(f"Error: {e}")
                    print("Please try again.")
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user")
        except EOFError:
            print("\n\nSession ended")


if __name__ == "__main__":
    bot = VanillaChatEngine(user_id="vanilla_test_user")
    bot.chat_loop()
