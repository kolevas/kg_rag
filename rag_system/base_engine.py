
from huggingface_hub import InferenceClient
import os
import json
import re
from dotenv import load_dotenv
from preprocessing.simplified_document_reader import DocumentReader
from chat_history.mongo_chat_history import MongoDBChatHistoryManager
from kg_retriever import KnowledgeGraphRetriever

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

_SYSTEM_PROMPT = """You are a knowledgeable AI assistant that can answer questions about literature, science, and various other topics. 
When answering:
- Use the provided context to give accurate, detailed responses. Be specific and informative in your responses
- If previous conversation history is available, reference it when relevant
- Use ONLY the provided context and conversation history to answer questions. If you don't know the answer state that.
- Generate follow-up question suggestions based on the context after each response to keep the conversation going
- Your responses should look like this:
```json
{{
    "response": "Your answer here",
    "follow_up_questions": [
        "Follow-up question 1",
        "Follow-up question 2",
        "Follow-up question 3"
    ]
}}
```
- Stick to the provided guideline and format
- If the question is vague or ambiguous, ask for clarification
Context:
{context}
History:
{history}
If by any chance you find conflicting information in the context, use the most recent information to answer the question."""

_DEFAULT_FOLLOW_UPS = [
    "Can you tell me more about this topic?",
    "What are the key points I should know?",
    "Are there any related topics I should explore?",
]

class BaseChatEngine:
    _CHROMA_DB_PATH = "./chroma_db"
    _KG_PATH = "./knowledge_graph"
    _CHAT_HISTORY_IMPORT = None 

    def __init__(self, user_id="default_user"):
        self.user_id = user_id
        self.project_id = "1"
        self.reader = DocumentReader(chroma_db_path=self._CHROMA_DB_PATH)
        self.chat_manager = MongoDBChatHistoryManager(db_name="chat_history_db", collection_name="conversations")
        self.session_id = self._initialize_session()
        self.kg_retriever = self._initialize_kg_retriever()
        self.client, self.model_name = self._initialize_llm()

    def _initialize_session(self):
        sessions = self.chat_manager.get_sessions(self.user_id, self.project_id)
        if sessions:
            return sessions[-1]["session_id"]
        else:
            session = self.chat_manager.create_session(self.user_id, self.project_id)
            return session["session_id"]

    def _initialize_kg_retriever(self):
        try:
            retriever = KnowledgeGraphRetriever(kg_path=self._KG_PATH)
            return retriever
        except Exception as e:
            print(f"Could not initialize Knowledge Graph: {e}")
            return None

    def _initialize_llm(self):
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment variables.")

        client = InferenceClient(token=hf_token)
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        print(f"Using Hugging Face Inference API with model: {model_name}")
        return client, model_name

    def get_sessions(self):
        try:
            sessions = self.chat_manager.get_sessions(self.user_id, self.project_id)
            return sessions if sessions else []
        except Exception as e:
            print(f"Error retrieving sessions: {e}")
            return []


    def _get_conversation_history(self, query: str):
        history = self.chat_manager.find_most_relevant_conversation(
            query, session_id=self.session_id, n_results=5, max_tokens=500)
        if not history:
            print("No relevant conversation history found")
            return None

        if isinstance(history, list):
            print(f"Retrieved {len(history)} relevant conversation(s)")
            return history[0] if history else None   #check

        count = max(history.count("Question:"), 1)
        print(f"Retrieved {count} relevant conversation(s)")
        return history

    def _get_document_context(self, query: str) -> str:  #token count limit missing; add token count limit and concat chunks until limit is reached
        print("Retrieving documents")
        retrieved_content = self.reader.query_documents(
            query=query,
            collection_name="multimodal_data",
            n_results=20,
        )
        context_string = "\n".join(retrieved_content[:10])
        print(f"Using {len(retrieved_content[:10])} document chunks")
        return context_string

    def _get_kg_context(self, query: str) -> str: #max triples shouldnt be static; consider dynamic limit based on token count of KG triples and document context
        """Retrieve and process knowledge graph context."""
        if not self.kg_retriever:
            return ""
        try:
            kg_context = self.kg_retriever.retrieve_kg_context(query, max_triples=8) 
            if kg_context:
                print("Retrieved KG context")
            return kg_context
        except Exception as e:
            print(f"Error retrieving KG context: {e}")
            return ""

    def _build_system_message(self, context_string: str, kg_context: str = "", history_text: str = "") -> str:
        context = f"{kg_context}\n\n{context_string}" if kg_context else context_string
        return _SYSTEM_PROMPT.format(context=context, history=history_text)

    def _improve_query(self, query: str, is_interactive: bool = False) -> str:
        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "You are an assistant that helps improve search queries. "
                        "Rewrite this question to be more specific and helpful for "
                        "document search. Only output the improved question, nothing "
                        f"else.\n\nOriginal: {query}\n\nImproved:"
                    ),
                }
            ]

            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=60,
                temperature=0.3,
                top_p=0.9,
            )

            improved = response.choices[0].message.content.strip()
            if improved and improved.lower() != query.lower():
                print(f"Improved query: {improved}")
                return improved 
        except Exception as e:
            print(f"Error improving query: {type(e).__name__}: {str(e)}")
        return query

    @staticmethod
    def _wrap_as_json(text: str, follow_ups: list | None = None) -> str:
        return json.dumps(
            {
                "response": text,
                "follow_up_questions": follow_ups or _DEFAULT_FOLLOW_UPS,
            },
            indent=2,
        )

    def _generate_response(self, query: str, system_message: str) -> str:
        print("Generating response")
        improved_query = self._improve_query(query)

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": improved_query},
                ],
                model=self.model_name,
                max_tokens=2000,
                temperature=0.6,
                top_p=0.9,
            )

            text = response.choices[0].message.content.strip()

            try:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match and "response" in json.loads(match.group()):
                    return text
            except (json.JSONDecodeError, ValueError):
                pass
            return self._wrap_as_json(text)

        except Exception as e:
            print(f"Error generating response: {type(e).__name__}: {str(e)}")
            return self._wrap_as_json(
                "I apologize, but I encountered an error while generating a response. "
                "Please try rephrasing your question."
            )

    def _save_conversation(self, query: str, response: str):
        self.chat_manager.add_conversation(query=query, response=response, session_id=self.session_id)

    def chat_loop(self):
        print("Document Chat Bot initialized!")
        print(f"LLM: {self.model_name}")
        print("Commands: 'quit'")
        print("-" * 60)
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
                    response = self.process_query(query)
                    print("\nResponse:")
                    print(response)
                    print("-" * 60)
                except Exception as e:
                    print(f"Error: {e}")
                    print("Please try again.")
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user")
        except EOFError:
            print("\n\nSession ended")

    def process_query(self, query: str) -> str:
        relevant_history = self._get_conversation_history(query)
        context_string = self._get_document_context(query)
        kg_context = self._get_kg_context(query)
        history_text = ""
        if relevant_history:
            history_text = f"\n\nRelevant Context:\n{relevant_history}"

        system_message = self._build_system_message(
            context_string, kg_context, history_text
        )
        response = self._generate_response(query, system_message)
        self._save_conversation(query, response)
        return response
