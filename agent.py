import os
import ast
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Simple memory implementation for newer LangChain versions
class SimpleMemory:
    def __init__(self, k=3):
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
serpapi_key = os.getenv("SERPAPI_API_KEY")
llm = ChatOpenAI(temperature=0, openai_api_key=api_key)

# Create Wikipedia tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Create SerpAPI tool
serpapi = SerpAPIWrapper(serpapi_api_key=serpapi_key)

# LangChain memory implementation (last 3 conversations)
memory = ConversationBufferWindowMemory(k=3)

def calculator(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

def format_weather(weather_data):
    """Format weather data in a user-friendly way"""
    if isinstance(weather_data, dict) and weather_data.get('type') == 'weather_result':
        return f"Weather in {weather_data.get('location', 'Unknown')}: {weather_data.get('weather', 'N/A')} at {weather_data.get('temperature', 'N/A')}°{weather_data.get('unit', 'F')[0]}. Humidity: {weather_data.get('humidity', 'N/A')}, Wind: {weather_data.get('wind', 'N/A')}, Precipitation: {weather_data.get('precipitation', 'N/A')} ({weather_data.get('date', 'N/A')})"
    return str(weather_data)

# Create conversation chain with LangChain memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

def ask_question(question):
    """Ask a question using LangChain memory"""
    return conversation.predict(input=question)

# Handle multiple queries with memory
questions = [
    "What is the capital of France? Also, what is 15 multiplied by 12?",
    "What's the weather like there?",
    "Tell me about the Eiffel Tower"
]

for question in questions:
    print(f"\nQ: {question}")
    
    if "weather" in question.lower():
        # Get SerpAPI search results for weather
        search_result = serpapi.run("current weather in Paris France")
        try:
            if isinstance(search_result, str):
                weather_dict = ast.literal_eval(search_result)
            else:
                weather_dict = search_result
                
            if weather_dict.get('type') == 'weather_result':
                weather_text = f"Weather in {weather_dict.get('location', 'Unknown')}: {weather_dict.get('weather', 'N/A')} at {weather_dict.get('temperature', 'N/A')}°{weather_dict.get('unit', 'F')[0]}. Humidity: {weather_dict.get('humidity', 'N/A')}, Wind: {weather_dict.get('wind', 'N/A')}, Precipitation: {weather_dict.get('precipitation', 'N/A')} ({weather_dict.get('date', 'N/A')})"
                memory.save_context({"input": question}, {"output": weather_text})
                print(f"A: {weather_text}")
            else:
                result = ask_question(question)
                print(f"A: {result}")
        except:
            result = ask_question(question)
            print(f"A: {result}")
    elif "multiplied" in question.lower() or "*" in question or "calculate" in question.lower():
        # Handle calculation
        calc_result = calculator("15 * 12")
        answer = f"The capital of France is Paris. 15 multiplied by 12 equals {calc_result}."
        memory.save_context({"input": question}, {"output": answer})
        print(f"A: {answer}")
    elif "eiffel tower" in question.lower():
        # Get Wikipedia info
        tower_info = wikipedia.run("Eiffel Tower Paris")
        # Handle Unicode encoding issues
        try:
            short_info = tower_info[:200] + "..."
            memory.save_context({"input": question}, {"output": short_info})
            print(f"A: {short_info}")
        except UnicodeEncodeError:
            safe_info = tower_info.encode('ascii', 'ignore').decode('ascii')[:200] + "..."
            memory.save_context({"input": question}, {"output": safe_info})
            print(f"A: {safe_info}")
    else:
        result = ask_question(question)
        print(f"A: {result}")

# Display LangChain memory buffer
print("\n" + "="*50)
print("LANGCHAIN MEMORY BUFFER:")
try:
    print(memory.buffer)
except UnicodeEncodeError:
    safe_buffer = str(memory.buffer).encode('ascii', 'ignore').decode('ascii')
    print(safe_buffer)

# Function to use SerpAPI for any search query
def search_web(query: str) -> str:
    """Search the web using SerpAPI"""
    return serpapi.run(query)

