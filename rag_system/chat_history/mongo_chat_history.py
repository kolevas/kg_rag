import pymongo
from pymongo import MongoClient
import time
from typing import List, Dict, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
from token_utils import count_tokens  # Changed from relative import
from openai import AzureOpenAI
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables once at module level
load_dotenv()

class MongoDBChatHistoryManager:
    def create_session(self, user_id, project_id, title=None):
        import uuid, time
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "project_id": project_id,
            "created_at": time.time(),
            "title": title or f"Session {session_id}"
        }
        ##print(f"[DEBUG] Creating session: {session}")
        self.save_session(session)
        return session
    # --- Session Management Methods ---
    def save_session(self, session_dict):
        ##print(f"[DEBUG] Saving session to DB: {session_dict}")
        self.db['sessions'].insert_one(session_dict)

    def get_sessions(self, user_id, project_id):
        sessions = list(self.db['sessions'].find({'user_id': user_id, 'project_id': project_id}))
        ##print(f"[DEBUG] Retrieved sessions for user_id={user_id}, project_id={project_id}: {sessions}")
        return sessions

    def get_session(self, session_id, user_id, project_id):
        session = self.db['sessions'].find_one({'session_id': session_id, 'user_id': user_id, 'project_id': project_id})
        ##print(f"[DEBUG] get_session(session_id={session_id}, user_id={user_id}, project_id={project_id}) => {session}")
        return session

    def delete_session(self, session_id, user_id, project_id):
        result = self.db['sessions'].delete_one({'session_id': session_id, 'user_id': user_id, 'project_id': project_id})
        ##print(f"[DEBUG] delete_session(session_id={session_id}, user_id={user_id}, project_id={project_id}) => deleted_count={result.deleted_count}")
    def __init__(self, mongo_uri=None, db_name="chat_history_db", collection_name="conversations"):
        self.mongo_uri = mongo_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        self.db_name = db_name
        self.collection_name = collection_name
        
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        
        try:
            self.openai_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION")
            )
        except Exception as e:
            self.openai_client = None
        
        # Test connection and setup
        self._test_connection()
        self._setup_collection()
    
    def _test_connection(self):
        """Test MongoDB connection"""
        try:
            # Ping the database
            self.client.admin.command('ping')
        except Exception as e:
            #print(f"MongoDB connection failed: {e}")
            raise
    
    def _setup_collection(self):
        """Setup collection with proper indexes"""
        try:
            # Create indexes for better performance
            self.collection.create_index([("session_id", 1), ("timestamp", -1)])
            self.collection.create_index([("session_id", 1), ("type", 1)])
            self.collection.create_index([("timestamp", -1)])
            
        except Exception as e:
            pass  # Silently ignore index creation errors
   
    def _normalize_conversation(self, query: str, response: str) -> str:
        normalized_response = re.sub(r"\[\[\d+\]\],?\s*", "", response)
        normalized_response = re.sub(r"\n\s*#{1,6}\s*([^\n]+)", r". \1:", normalized_response)
        normalized_response = re.sub(r"\n\s*-\s*\*\*([^*]+)\*\*:\s*", r". \1: ", normalized_response)
        normalized_response = re.sub(r"\n\s*-\s*", ". ", normalized_response)
        normalized_response = re.sub(r"\*\*([^*]+)\*\*", r"\1", normalized_response)
        normalized_response = re.sub(r"\*([^*]+)\*", r"\1", normalized_response)
        normalized_response = re.sub(r"\n{3,}", "\n\n", normalized_response)
        normalized_response = re.sub(r"\n\n", ". ", normalized_response)
        normalized_response = re.sub(r"\n", " ", normalized_response)
        normalized_response = re.sub(r"\.+", ".", normalized_response)
        normalized_response = re.sub(r"\s*\.\s*\.", ".", normalized_response)
        normalized_response = re.sub(r":\s*\.", ":", normalized_response)
        normalized_response = re.sub(r"\.\s*:", ":", normalized_response)
        normalized_response = re.sub(r"\s+", " ", normalized_response)
        normalized_response = re.sub(r"\s*([.,:;!?])", r"\1", normalized_response)
        normalized_response = re.sub(r"([.,:;!?])\s*", r"\1 ", normalized_response)
        normalized_response = re.sub(r"\.\s*([a-z])", lambda m: ". " + m.group(1).upper(), normalized_response)
        normalized_response = re.sub(r"(\d+)\.\s+(\d+)", r"\1.\2", normalized_response)
        normalized_response = normalized_response.strip()
        if normalized_response and not normalized_response[0].isupper():
            normalized_response = normalized_response[0].upper() + normalized_response[1:]
        if normalized_response and normalized_response[-1] not in '.!?':
            normalized_response += "."
        return normalized_response
    
    def _summarize_conversation(self, query: str, response: str) -> str:
        """Summarize conversation using OpenAI model"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are an expert at creating concise summaries while preserving ALL important information. Create a summary that: 
                        1) Retains all key facts, numbers, data points, and specific details 
                        2) Maintains the logical flow of the conversation 
                        3) Targets 60% of original length while preserving 100% of factual content 
                        4) Returns clean text without markdown and new lines (don't add \n)
                        5) Prioritizes factual accuracy over brevity"""}, 
                    {"role": "user", "content": f"Please summarize this conversation:\nUser: {query}\nAssistant: {response}"}
                ],
                max_tokens=500,
                temperature=0.3,
                timeout=30
            )
            summary = response.choices[0].message.content.strip()
            if summary and len(summary) > 10:
                return summary
        except Exception as e:
            return self._normalize_conversation(query, response)
    def add_conversation(self, query: str, response: str, session_id: str, timestamp: float = None):
        """Add a conversation pair to history for a specific session"""
        if timestamp is None:
            timestamp = time.time()
            
        document = {
            "_id": f"{session_id}_conv_{int(timestamp * 1000)}",
            "query": query,
            "response": response,
            "session_id": session_id,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp),
            "type": "conversation",
            "summary": self._summarize_conversation(query, response) if self.openai_client else self._normalize_conversation(query, response)
        }
        
        #print(f"[DEBUG] Saving conversation with document ID: {document['_id']}")
        #print(f"[DEBUG] Session ID: {session_id}")
        #print(f"[DEBUG] Query: {query[:100]}...")
        #print(f"[DEBUG] Response: {response[:100]}...")
        
        try:
            result = self.collection.replace_one(
                {"_id": document["_id"]},
                document,
                upsert=True
            )
            #print(f"[DEBUG] Save result - matched: {result.matched_count}, modified: {result.modified_count}, upserted: {result.upserted_id}")
            
            # Verify the document was saved
            saved_doc = self.collection.find_one({"_id": document["_id"]})
                
        except Exception as e:
            #print(f"[DEBUG] Error adding conversation: {e}")
            import traceback
            traceback.print_exc()
            raise

    def find_most_relevant_conversation(self, current_query: str, session_id: str, n_results: int = 5, max_tokens: int = 2000) -> Optional[str]:
        """Find the most relevant conversations using only the hybrid approach."""
        conversations = self.get_hybrid_conversations(current_query, session_id, n_results, max_tokens)
        if not conversations:
            return None
        context_parts = []
        for conv in conversations:
            context_parts.append(f"[CONVERSATION] {conv.get('summary', '')}")
        return "\n".join(context_parts)

    def get_hybrid_conversations(self, current_query: str, session_id: str, n_results: int = 5, max_tokens: int = 3000) -> List[Dict]: 
        """
        Simplified hybrid approach: 30% recent, 70% similarity-based retrieval
        """
        try:
            # Check if conversations exist
            total_conversations = self.collection.count_documents({"session_id": session_id, "type": "conversation"})
            if total_conversations == 0:
                return []
            
            # Simple 30/70 split
            recent_tokens = int(max_tokens * 0.3)
            similarity_tokens = int(max_tokens * 0.7)
            
            # Get recent and similar conversations
            recent_conversations = self._get_recent_conversations_for_hybrid(session_id, recent_tokens)
            similarity_conversations = self._get_similarity_conversations(current_query, session_id, similarity_tokens, recent_conversations)
            
            # Combine without duplicates
            combined_conversations = recent_conversations.copy()
            recent_ids = {conv.get('_id', '') for conv in recent_conversations}
            
            for conv in similarity_conversations:
                if conv.get('_id', '') not in recent_ids:
                    combined_conversations.append(conv)
            
            return combined_conversations
            
        except Exception as e:
            #print(f"Error in hybrid conversation retrieval: {e}")
            return []
    
    def _get_recent_conversations_for_hybrid(self, session_id: str, max_tokens: int) -> List[Dict]:
        """Get recent conversations within token limit"""
        try:
            conversations = list(self.collection.find(
                {"session_id": session_id, "type": "conversation"}
            ).sort("timestamp", -1).limit(10))
            
            result = []
            total_tokens = 0
            
            for conv in conversations:
                summary_tokens = count_tokens(conv.get('summary', ''))
                if total_tokens + summary_tokens <= max_tokens:
                    result.append({
                        '_id': conv.get('_id'),
                        'summary': conv.get('summary', ''),
                        'timestamp': conv['timestamp'],
                        'source': 'recent'
                    })
                    total_tokens += summary_tokens
                else:
                    break
            
            return result
            
        except Exception as e:
            #print(f"Error getting recent conversations: {e}")
            return []
    
    def _get_similarity_conversations(self, current_query: str, session_id: str, max_tokens: int, exclude_conversations: List[Dict]) -> List[Dict]:
        """Simplified: Get most similar conversations to current query using TF-IDF cosine similarity."""
        try:
            all_convs = list(self.collection.find({"session_id": session_id, "type": "conversation"}).sort("timestamp", -1))
            excluded_ids = {conv.get('_id', '') for conv in exclude_conversations}
            candidates = [conv for conv in all_convs if conv.get('_id', '') not in excluded_ids]
            if not candidates:
                return []
            texts = [f"{conv['query']} {conv['response']}" for conv in candidates]
            vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
            tfidf = vectorizer.fit_transform([current_query] + texts)
            sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
            sorted_idx = np.argsort(sims)[::-1]
            result, total_tokens = [], 0
            for idx in sorted_idx:
                conv = candidates[idx]
                summary = conv.get('summary', '')
                tokens = count_tokens(summary)
                if total_tokens + tokens > max_tokens:
                    break
                result.append({
                    '_id': conv.get('_id'),
                    'summary': summary,
                    'timestamp': conv['timestamp'],
                    'similarity_score': float(sims[idx]),
                    'source': 'similarity'
                })
                total_tokens += tokens
            return result
        except Exception as e:
            #print(f"Error in similarity search: {e}")
            return []

    def clear_history(self, session_id: str = None):
        """Clear chat history for a session or all if session_id is None"""
        try:
            if session_id:
                result = self.collection.delete_many({"session_id": session_id})
                #print(f"Chat history cleared for session '{session_id}' - {result.deleted_count} documents deleted")
            else:
                result = self.collection.delete_many({})
                #print(f"Chat history cleared for all sessions - {result.deleted_count} documents deleted")
        except Exception as e:
            print(f"Error clearing history: {e}")
 
    def get_conversation(self, session_id: str) -> List[tuple]:
        """Get all conversations for a session as a list of (role, content) tuples"""
        try:
            #print(f"[DEBUG] Fetching conversations for session_id: {session_id}")
            
            # First, let's see what's in the database for this session
            all_docs = list(self.collection.find({"session_id": session_id}))
            #print(f"[DEBUG] Found {len(all_docs)} total documents for session {session_id}")
            
            # Now get conversations specifically
            conversations = list(self.collection.find(
                {"session_id": session_id, "type": "conversation"}
            ).sort("timestamp", 1))  # Sort by timestamp ascending (oldest first)
            
            #print(f"[DEBUG] Found {len(conversations)} conversation documents")
            
            messages = []
            for i, conv in enumerate(conversations):
                #print(f"[DEBUG] Conversation {i}: query='{conv.get('query', '')[:50]}...', response='{conv.get('response', '')[:50]}...'")
                # Add user message
                messages.append(("user", conv.get("query", "")))
                # Add bot response  
                messages.append(("bot", conv.get("response", "")))
            
            #print(f"[DEBUG] Returning {len(messages)} messages")
            return messages
            
        except Exception as e:
            #print(f"[DEBUG] Error retrieving conversation for session {session_id}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def close(self):
        """Close MongoDB connection"""
        try:
            self.client.close()
        except Exception as e:
            pass  # Silently ignore close errors
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        try:
            self.close()
        except:
            pass
