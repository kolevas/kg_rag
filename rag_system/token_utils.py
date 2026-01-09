import tiktoken

class TokenUtils:
    """Utility class for token counting and management"""
    
    def __init__(self):
        # Use tiktoken for OpenAI-compatible token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4/ChatGPT tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum token count"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)

# Legacy functions for backward compatibility
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))
