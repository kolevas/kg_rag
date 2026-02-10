import uuid
import time
import re
import os
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from token_utils import count_tokens

load_dotenv()

_NORMALIZE_PIPELINE = [
    (re.compile(r"\[\[\d+\]\],?\s*"), ""),                      
    (re.compile(r"\n\s*#{1,6}\s*([^\n]+)"), r". \1:"),           
    (re.compile(r"\n\s*-\s*\*\*([^*]+)\*\*:\s*"), r". \1: "),  
    (re.compile(r"\n\s*-\s*"), ". "),                            
    (re.compile(r"\*{1,2}([^*]+)\*{1,2}"), r"\1"),           
    (re.compile(r"\n+"), ". "),                             
    (re.compile(r"\.[\s.]+"), ". "),                           
    (re.compile(r":\s*\."), ":"),                               
    (re.compile(r"\.\s*:"), ":"),                          
    (re.compile(r"\s+"), " "),                                  
    (re.compile(r"\s*([.,:;!?])"), r"\1"),                   
    (re.compile(r"([.,:;!?])(?!\s)"), r"\1 "),         
    (re.compile(r"\.\s*([a-z])"), lambda m: ". " + m.group(1).upper()),
    (re.compile(r"(\d+)\.\s+(\d+)"), r"\1.\2"),                
]


class MongoDBChatHistoryManager:

    def __init__(self, mongo_uri=None, db_name="chat_history_db", collection_name="conversations"):
        self.mongo_uri = mongo_uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.hf_client = InferenceClient(token=hf_token) if hf_token else None
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"

        # Verify connection
        self.client.admin.command('ping')

        # Create indexes for query performance
        self.collection.create_index([("session_id", 1), ("timestamp", -1)])
        self.collection.create_index([("session_id", 1), ("type", 1)])
        self.collection.create_index([("timestamp", -1)])

    def create_session(self, user_id, project_id, title=None):
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "project_id": project_id,
            "created_at": time.time(),
            "title": title or f"Session {session_id}",
        }
        self.db['sessions'].insert_one(session)
        return session

    def get_sessions(self, user_id, project_id):
        return list(self.db['sessions'].find({'user_id': user_id, 'project_id': project_id}))

    def get_session(self, session_id, user_id, project_id):
        return self.db['sessions'].find_one({
            'session_id': session_id, 'user_id': user_id, 'project_id': project_id
        })

    def delete_session(self, session_id, user_id, project_id):
        self.db['sessions'].delete_one({
            'session_id': session_id, 'user_id': user_id, 'project_id': project_id
        })

    def add_conversation(self, query: str, response: str, session_id: str, timestamp: float = None):
        timestamp = timestamp or time.time()
        summary = (
            self._summarize_conversation(query, response)
            if self.hf_client
            else self._normalize_conversation(query, response)
        )
        document = {
            "_id": f"{session_id}_conv_{int(timestamp * 1000)}",
            "query": query,
            "response": response,
            "session_id": session_id,
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp),
            "type": "conversation",
            "summary": summary,
        }
        self.collection.replace_one({"_id": document["_id"]}, document, upsert=True)

    def get_conversation(self, session_id: str) -> List[tuple]:
        conversations = list(self.collection.find(
            {"session_id": session_id, "type": "conversation"}
        ).sort("timestamp", 1))
        messages = []
        for conv in conversations:
            messages.append(("user", conv.get("query", "")))
            messages.append(("bot", conv.get("response", "")))
        return messages

    def clear_history(self, session_id: str = None):
        query = {"session_id": session_id} if session_id else {}
        self.collection.delete_many(query)

    def find_most_relevant_conversation(
        self, current_query: str, session_id: str,
        n_results: int = 5, max_tokens: int = 2000,
    ) -> Optional[str]:
        all_convs = list(self.collection.find(
            {"session_id": session_id, "type": "conversation"}
        ).sort("timestamp", -1))
        if not all_convs:
            return None

        recent_tokens = int(max_tokens * 0.3)
        similarity_tokens = int(max_tokens * 0.7)

        recent, used_ids, tokens_used = [], set(), 0
        for conv in all_convs:
            t = count_tokens(conv.get('summary', ''))
            if tokens_used + t > recent_tokens:
                break
            recent.append(conv)
            used_ids.add(conv['_id'])
            tokens_used += t

        candidates = [c for c in all_convs if c['_id'] not in used_ids]
        similar = self._rank_by_similarity(current_query, candidates, similarity_tokens)

        parts = [f"[CONVERSATION] {c.get('summary', '')}" for c in recent + similar]
        return "\n".join(parts) if parts else None

    def _rank_by_similarity(
        self, query: str, candidates: List[Dict], max_tokens: int,
    ) -> List[Dict]:
        if not candidates:
            return []
        texts = [f"{c['query']} {c['response']}" for c in candidates]
        tfidf = TfidfVectorizer(stop_words='english', max_features=300).fit_transform([query] + texts)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:])[0]

        result, tokens_used = [], 0
        for idx in np.argsort(scores)[::-1]:
            t = count_tokens(candidates[idx].get('summary', ''))
            if tokens_used + t > max_tokens:
                break
            result.append(candidates[idx])
            tokens_used += t
        return result

    def _normalize_conversation(self, query: str, response: str) -> str:
        """Strip markdown and clean response into plain-text sentences."""
        text = response
        for pattern, repl in _NORMALIZE_PIPELINE:
            text = pattern.sub(repl, text)
        text = text.strip()
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in '.!?':
            text += "."
        return text

    def _summarize_conversation(self, query: str, response: str) -> str:
        """Summarize a conversation turn via the HuggingFace Inference API."""
        try:
            result = self.hf_client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": (
                    "You are an expert at creating concise summaries while preserving ALL important information. "
                    "Create a summary that: "
                    "1) Retains all key facts, numbers, data points, and specific details. "
                    "2) Maintains the logical flow of the conversation. "
                    "3) Targets 60% of original length while preserving 100% of factual content. "
                    "4) Returns clean text without markdown and new lines. "
                    "5) Prioritizes factual accuracy over brevity.\n\n"
                    f"Please summarize this conversation:\nUser: {query}\nAssistant: {response}"
                )}],
                max_tokens=500,
                temperature=0.3,
            )
            summary = result.choices[0].message.content.strip()
            if summary and len(summary) > 10:
                return summary
        except Exception:
            pass
        return self._normalize_conversation(query, response)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
