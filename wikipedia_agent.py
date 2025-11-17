import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

# Initialize LLM
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, openai_api_key=api_key)

# Create Wikipedia tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information"""
    try:
        result = wikipedia.run(query)
        return result[:500] + "..." if len(result) > 500 else result
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

def ask_with_wikipedia(question: str) -> str:
    """Answer questions using Wikipedia search"""
    # First try to get Wikipedia info
    wiki_info = search_wikipedia(question)
    
    # Use LLM to provide a better answer based on Wikipedia info
    prompt = f"Based on this Wikipedia information: {wiki_info}\n\nAnswer this question: {question}"
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    questions = [
        "Tell me about the Eiffel Tower",
        "What is machine learning?",
        "Who was Albert Einstein?"
    ]
    
    print("=== Wikipedia Agent ===")
    for question in questions:
        print(f"\nQ: {question}")
        answer = ask_with_wikipedia(question)
        print(f"A: {answer}")