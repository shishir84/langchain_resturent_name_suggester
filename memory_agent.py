import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Simple memory implementation
class SimpleMemory:
    def __init__(self, k=5):
        self.k = k
        self.conversations = []
    
    def save_context(self, inputs, outputs):
        self.conversations.append({"input": inputs["input"], "output": outputs["output"]})
        if len(self.conversations) > self.k:
            self.conversations.pop(0)
    
    @property
    def buffer(self):
        return "\n".join([f"Human: {conv['input']}\nAI: {conv['output']}" for conv in self.conversations])
    
    def get_context(self):
        return self.buffer

load_dotenv()

# Initialize LLM
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, openai_api_key=api_key)

# Initialize memory
memory = SimpleMemory(k=5)

def ask_with_memory(question: str) -> str:
    """Ask a question with conversation memory"""
    context = memory.get_context()
    if context:
        prompt = f"Previous conversation:\n{context}\n\nHuman: {question}\nAI:"
    else:
        prompt = f"Human: {question}\nAI:"
    
    response = llm.invoke(prompt)
    answer = response.content
    
    # Save to memory
    memory.save_context({"input": question}, {"output": answer})
    return answer

def show_memory():
    """Display current memory buffer"""
    print("\n" + "="*50)
    print("CONVERSATION HISTORY:")
    print(memory.buffer if memory.buffer else "No conversation history yet.")
    print("="*50)

if __name__ == "__main__":
    print("=== Memory Agent ===")
    print("This agent remembers our conversation!")
    
    questions = [
        "My name is John and I like pizza",
        "What's my name?",
        "What food do I like?",
        "Can you remember what we talked about?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = ask_with_memory(question)
        print(f"A: {answer}")
    
    show_memory()