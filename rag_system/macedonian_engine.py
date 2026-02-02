from huggingface_hub import InferenceClient
import deepl
# Use package-qualified imports so the module can be imported from the app root
from rag_system.preprocessing.document_reader import DocumentReader
from rag_system.kg_retriever import KnowledgeGraphRetriever
import os
import json
import re
from dotenv import load_dotenv
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

class MacedonianChatBot:

    def __init__(self, user_id="default_user"):
        self.user_id = user_id
        self.project_id = "1"
        self.reader = DocumentReader(chroma_db_path="./chroma_db")
        from rag_system.chat_history.mongo_chat_history import MongoDBChatHistoryManager
        self.chat_manager = MongoDBChatHistoryManager(db_name="chat_history_db", collection_name="conversations")
        self.session_id = self._initialize_session()
        
        # Initialize Knowledge Graph Retriever
        print("Initializing Knowledge Graph Retriever...")
        try:
            self.kg_retriever = KnowledgeGraphRetriever(kg_path="./rag_system/knowledge_graph")
            print("✅ Knowledge Graph loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not initialize Knowledge Graph: {e}")
            self.kg_retriever = None
        
        # Initialize Hugging Face Inference API client
        print("Initializing Hugging Face Inference API client...")
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise ValueError(
                "⚠️ HUGGINGFACE_TOKEN not found in environment variables.\n"
                "Please add your Hugging Face token to the .env file:\n"
                "HUGGINGFACE_TOKEN=your_token_here"
            )
        
        self.client = InferenceClient(token=hf_token)
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        print(f"✅ Using Hugging Face Inference API with model: {self.model_name}")
        
        # Initialize DeepL translator
        print("Initializing DeepL translator...")
        self.translation_enabled = self._init_deepl_translator()
        if self.translation_enabled:
            print("✅ DeepL translator initialized successfully!")
        else:
            print("⚠️ DeepL translator failed to initialize. Will use English-only mode.")

    def _init_deepl_translator(self) -> bool:
        """Initialize DeepL translator. Returns True if successful."""
        try:
            deepl_key = os.getenv("DEEPL_API_KEY")
            if not deepl_key:
                print("⚠️ DEEPL_API_KEY not found in environment variables.")
                print("Please add your DeepL API key to the .env file:")
                print("DEEPL_API_KEY=your_api_key_here")
                print("Get a free API key at: https://www.deepl.com/pro-api")
                return False
            
            self.translator = deepl.Translator(deepl_key)
            
            # Test the connection
            usage = self.translator.get_usage()
            if usage.character.limit_exceeded:
                print("⚠️ DeepL character limit exceeded!")
                return False
            
            print(f"📊 DeepL usage: {usage.character.count}/{usage.character.limit} characters")
            return True
        except Exception as e:
            print(f"⚠️ Error initializing DeepL: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def translate_mk_to_en(self, text: str) -> str:
        """Translate Macedonian text to English using DeepL."""
        if not self.translation_enabled:
            print("⚠️ Translation not available, returning original text")
            return text
        
        try:
            print(f"🔄 Translating MK→EN: '{text[:50]}...'")
            result = self.translator.translate_text(text, source_lang="MK", target_lang="EN-US")
            translated_text = result.text
            print(f"✅ Translation result: '{translated_text[:50]}...'")
            return translated_text
        except Exception as e:
            print(f"⚠️ Translation error (mk→en): {e}")
            import traceback
            traceback.print_exc()
            return text
    
    def translate_en_to_mk(self, text: str) -> str:
        """Translate English text to Macedonian using DeepL."""
        if not self.translation_enabled:
            return text
        
        try:
            result = self.translator.translate_text(text, source_lang="EN", target_lang="MK")
            translated_text = result.text
            return translated_text
        except Exception as e:
            print(f"⚠️ Translation error (en→mk): {e}")
            import traceback
            traceback.print_exc()
            return text
    
    def is_macedonian(self, text: str) -> bool:
        """Check if text contains Macedonian Cyrillic characters."""
        macedonian_chars = re.findall(r'[АБВГДЃЕЖЗЅИЈКЛЉМНЊОПРСТЌУФХЦЧЏШабвгдѓежзѕијклљмнњопрстќуфхцчџш]', text)
        return len(macedonian_chars) > 10

    def _initialize_session(self):
        # Use chat_manager to create or get a session (MongoDB only)
        sessions = self.chat_manager.get_sessions(self.user_id, self.project_id)
        if sessions:
            return sessions[-1]["session_id"]  # Use the most recent session
        else:
            session = self.chat_manager.create_session(self.user_id, self.project_id)
            return session["session_id"]

    def _initialize_chat_manager(self):
        """Initialize the MongoDB chat history manager only."""
        from rag_system.chat_history.mongo_chat_history import MongoDBChatHistoryManager
        return MongoDBChatHistoryManager(
            db_name="chat_history_db", collection_name="conversations"
        )
    
    def get_sessions(self):
        """Return all sessions for the current user as a list of dicts."""
        try:
            sessions = self.chat_manager.get_sessions(self.user_id, self.project_id)
            return sessions if sessions else []
        except Exception as e:
            print(f"Error retrieving sessions: {e}")
            return []

    def _get_conversation_history(self, query: str):
        """Get relevant conversation history based on storage type."""
        print("Searching for relevant conversation history...")
        relevant_history = self.chat_manager.find_most_relevant_conversation(
            query, session_id=self.session_id, n_results=5, max_tokens=500
        )
        if relevant_history:
            if isinstance(relevant_history, list):
                conversation_count = len(relevant_history)
                return relevant_history[0] if relevant_history else None
            else:
                conversation_count = relevant_history.count("Question:")
                if conversation_count == 0:
                    conversation_count = 1  # Assume at least one conversation if content exists
            
            print(f"📊 Retrieved {conversation_count} relevant conversation(s)")
            return relevant_history
        else:
            print("📊 No relevant conversation history found")
            return None
    
    def _get_document_context(self, query: str) -> str:
        """Retrieve and process document context."""
        print(f"Retrieving documents...")
        retrieved_content = self.reader.query_documents(
            query=query,
            collection_name="multimodal_data",
            n_results=20
        )
        
        context_string = "\n".join(retrieved_content[:10])  # Use top 10 chunks
        print(f"Using {len(retrieved_content[:10])} document chunks")
        return context_string
    
    def _get_kg_context(self, query: str) -> str:
        """Retrieve and process knowledge graph context."""
        if not self.kg_retriever:
            return ""
        
        try:
            print("Retrieving from knowledge graph...")
            kg_context = self.kg_retriever.retrieve_kg_context(query, max_triples=8)
            if kg_context:
                print(f"✅ Retrieved KG context")
            return kg_context
        except Exception as e:
            print(f"⚠️ Error retrieving KG context: {e}")
            return ""
    
    def _build_system_message(self, context_string: str, kg_context: str = "", history_text: str = "") -> str:
        """Build the system message for the LLM."""
        
        combined_context = context_string
        if kg_context:
            combined_context = f"{kg_context}\n\n{context_string}"
        
        return f"""You are a knowledgeable AI assistant that can answer questions about literature, science, and various other topics. 
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
                {combined_context}
                History:
                {history_text}
                If by any chance you find conflicting information in the context, use the most recent information to answer the question."""

    def _improve_query(self, query: str, is_interactive: bool = False) -> str:
        """Improve query using LLM. Query is expected to be in English."""
        try:
            # Generate improved query using Hugging Face Inference API with chat completion
            messages = [
                {
                    "role": "user",
                    "content": f"You are an assistant that helps improve search queries. Rewrite this question to be more specific and helpful for document search. Only output the improved question, nothing else.\n\nOriginal: {query}\n\nImproved:"
                }
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=60,
                temperature=0.3,
                top_p=0.9
            )
            
            improved = response.choices[0].message.content.strip()
            if improved and improved.lower() != query.lower():
                print(f"Improved query: {improved}")
                if is_interactive:
                    print("Would you like to use this improved query? (yes/no)")
                    user_choice = input().strip().lower()
                    if user_choice in ['yes', 'y']: 
                        return improved
                    return query  # Use original query if user declines
                else:
                    return improved
        except Exception as e:
            print(f"⚠️ Error improving query: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
        return query

    def _generate_response(self, query: str, system_message: str) -> str:
        """Generate response using Hugging Face Inference API. Query is expected to be in English."""
        print("Generating response...")
        improved_query = self._improve_query(query)
        
        try:
            # Generate response using Hugging Face Inference API with chat completion
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": improved_query}
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=2000,
                temperature=0.6,
                top_p=0.9
            )
            
            # Extract the actual content from the response
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response format
            try:
                # Look for JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_json = json.loads(json_str)
                    if "response" in parsed_json:
                        return response_text  # Return the full response with JSON
                else:
                    # If no JSON found, format the response manually
                    formatted_response = {
                        "response": response_text,
                        "follow_up_questions": [
                            "Can you tell me more about this topic?",
                            "What are the key points I should know?",
                            "Are there any related topics I should explore?"
                        ]
                    }
                    return json.dumps(formatted_response, indent=2)
            except json.JSONDecodeError:
                # If JSON parsing fails, format manually
                formatted_response = {
                    "response": response_text,
                    "follow_up_questions": [
                        "Can you provide more details?",
                        "What else should I know about this?",
                        "Are there related topics worth exploring?"
                    ]
                }
                return json.dumps(formatted_response, indent=2)
            
            return response_text
            
        except Exception as e:
            print(f"⚠️ Error generating response: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return json.dumps({
                "response": "I apologize, but I encountered an error while generating a response. Please try rephrasing your question.",
                "follow_up_questions": [
                    "Could you rephrase your question?",
                    "What specific aspect would you like to know about?",
                    "Is there a particular detail you're interested in?"
                ]
            }, indent=2)
    
    def process_query(self, query: str) -> str:
      
        original_query = query
        is_mk = self.is_macedonian(query)
        
        # Translate to English if Macedonian
        if is_mk:
            print("🌐 Detected Macedonian input, translating to English for reasoning...")
            query = self.translate_mk_to_en(query)
            print(f"📝 English query: {query}")
        
        # Get conversation history, document context, and KG context (in English)
        relevant_history = self._get_conversation_history(query)
        context_string = self._get_document_context(query)
        kg_context = self._get_kg_context(query)
        
        # Prepare context for LLM
        history_text = ""
        if relevant_history:
            history_text = f"\n\nRelevant Context:\n{relevant_history}"
        
        # Generate response (in English)
        system_message = self._build_system_message(context_string, kg_context, history_text)
        response = self._generate_response(query, system_message)
        
        # Translate response to Macedonian if original query was Macedonian
        if is_mk:
            print("🌐 Translating response to Macedonian...")
            try:
                # Parse JSON response
                response_data = json.loads(response)
                
                # Handle nested JSON string in response field
                if "response" in response_data and isinstance(response_data["response"], str):
                    # Try to parse nested JSON
                    try:
                        nested_json = json.loads(response_data["response"])
                        if "response" in nested_json:
                            nested_json["response"] = self.translate_en_to_mk(nested_json["response"])
                        if "follow_up_questions" in nested_json:
                            nested_json["follow_up_questions"] = [
                                self.translate_en_to_mk(q) for q in nested_json["follow_up_questions"]
                            ]
                        response_data["response"] = json.dumps(nested_json, ensure_ascii=False)
                    except json.JSONDecodeError:
                        # Not nested JSON, just translate the string
                        response_data["response"] = self.translate_en_to_mk(response_data["response"])
                
                # Translate follow-up questions at top level
                if "follow_up_questions" in response_data:
                    response_data["follow_up_questions"] = [
                        self.translate_en_to_mk(q) for q in response_data["follow_up_questions"]
                    ]
                
                response = json.dumps(response_data, indent=2, ensure_ascii=False)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing error: {e}")
                # If not JSON, translate the whole response
                response = self.translate_en_to_mk(response)
        
        # Save conversation (with original query and translated response)
        self._save_conversation(original_query, response)
        
        return response
    
    def _save_conversation(self, query: str, response: str):
        """Save conversation to chat history."""
        print("Saving conversation...")
        self.chat_manager.add_conversation(
            query=query, response=response, session_id=self.session_id
        )
    
    def chat_loop(self):
        """Main chat interaction loop with Macedonian translation support."""
        print(f"Macedonian Document Chat Bot initialized!")
        print(f"Storage: MongoDB")
        print(f"LLM: Mistral-7B (Hugging Face Inference API)")
        print(f"Translation: DeepL API (Macedonian ↔ English)")
        print(f"Retrieval: Documents + Knowledge Graph")
        print("Commands: 'quit'")
        print("-" * 60)
        try:
            while True:
                query = input("\nEnter your question (or command): ").strip()
                if not query:
                    continue
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                    
                try:
                    print(f"\n🔍 Processing: '{query}'")
                    response = self.process_query(query)
                    print("\nResponse:")
                    print(response)
                    print("-" * 60)
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Please try again.")
        except KeyboardInterrupt:
            print("\n\nSession interrupted by user")
        except EOFError:
            print("\n\nSession ended")

if __name__ == "__main__":
    bot = MacedonianChatBot(user_id="6175")
    bot.chat_loop()
