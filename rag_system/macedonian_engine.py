import deepl
import json
import os
import re
from base_engine import BaseChatEngine

class MacedonianChatBot(BaseChatEngine):

    _CHROMA_DB_PATH = "./chroma_db"
    _KG_PATH = "./knowledge_graph"

    def __init__(self, user_id="default_user"):
        super().__init__(user_id)
        self.translation_enabled = self._init_deepl_translator()
        if self.translation_enabled:
            print("DeepL translator initialized successfully!")
        else:
            print("DeepL translator failed to initialize. Will use English-only mode.")

    def _init_deepl_translator(self) -> bool:
        try:
            deepl_key = os.getenv("DEEPL_API_KEY")
            if not deepl_key:
                print("DEEPL_API_KEY not found in environment variables.")
                return False

            self.translator = deepl.Translator(deepl_key)

            usage = self.translator.get_usage()
            if usage.character.limit_exceeded:
                print("DeepL character limit exceeded!")
                return False

            print(f"DeepL usage: {usage.character.count}/{usage.character.limit} characters")
            return True
        except Exception as e:
            print(f"Error initializing DeepL: {type(e).__name__}: {str(e)}")
            return False

    def translate_mk_to_en(self, text: str) -> str:
        if not self.translation_enabled:
            print("Translation not available, returning original text")
            return text
        try:
            result = self.translator.translate_text(text, source_lang="MK", target_lang="EN-US")
            return result.text
        except Exception as e:
            print(f"Translation error (mk→en): {e}")
            return text

    def translate_en_to_mk(self, text: str) -> str:
        if not self.translation_enabled:
            return text
        try:
            result = self.translator.translate_text(text, source_lang="EN", target_lang="MK")
            return result.text
        except Exception as e:
            print(f"Translation error (en→mk): {e}")
            return text

    def is_macedonian(self, text: str) -> bool:
        macedonian_chars = re.findall(
            r'[АБВГДЃЕЖЗЅИЈКЛЉМНЊОПРСТЌУФХЦЧЏШабвгдѓежзѕијклљмнњопрстќуфхцчџш]',
            text,
        )
        return len(macedonian_chars) > 10

    def process_query(self, query: str) -> str:
        original_query = query
        is_mk = self.is_macedonian(query)

        if is_mk:
            query = self.translate_mk_to_en(query)
            print(f"English query: {query}")

        relevant_history = self._get_conversation_history(query)
        context_string = self._get_document_context(query)
        kg_context = self._get_kg_context(query)

        history_text = ""
        if relevant_history:
            history_text = f"\n\nRelevant Context:\n{relevant_history}"

        system_message = self._build_system_message(context_string, kg_context, history_text)
        response = self._generate_response(query, system_message)

        # Translate response back to Macedonian if needed
        if is_mk:
            response = self._translate_response_to_mk(response)

        self._save_conversation(original_query, response)
        return response

    def _translate_response_to_mk(self, response: str) -> str:
        print("Translating response to Macedonian")
        try:
            response_data = json.loads(response)

            if "response" in response_data and isinstance(response_data["response"], str):
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
                    response_data["response"] = self.translate_en_to_mk(response_data["response"])

            if "follow_up_questions" in response_data:
                response_data["follow_up_questions"] = [
                    self.translate_en_to_mk(q) for q in response_data["follow_up_questions"]
                ]

            return json.dumps(response_data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            return self.translate_en_to_mk(response)

    def chat_loop(self):
        """Chat loop with Macedonian translation support."""
        print("Macedonian Document Chat Bot initialized!")
        print(f"LLM: {self.model_name} ")
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
