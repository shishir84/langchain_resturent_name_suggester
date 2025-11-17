import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper

load_dotenv()

# Initialize LLM
api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")
llm = ChatOpenAI(temperature=0, openai_api_key=api_key)

# Create SerpAPI tool
serpapi = SerpAPIWrapper(serpapi_api_key=serpapi_key)

def search_web(query: str) -> str:
    """Search the web using SerpAPI"""
    try:
        result = serpapi.run(query)
        return str(result)
    except Exception as e:
        return f"Error searching web: {str(e)}"

def ask_with_search(question: str) -> str:
    """Answer questions using web search"""
    # Search the web for current information
    search_result = search_web(question)
    
    # Use LLM to provide a better answer based on search results
    prompt = f"Based on this web search result: {search_result}\n\nAnswer this question: {question}"
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    questions = [
        "What's the current weather in New York?",
        "Latest news about artificial intelligence",
        "Current stock price of Apple"
    ]
    
    print("=== SerpAPI Agent ===")
    for question in questions:
        print(f"\nQ: {question}")
        answer = ask_with_search(question)
        print(f"A: {answer}")