#!/usr/bin/env python3
"""
Interactive Demo for LangChain Multi-Agent System
Run this script to test all agents interactively
"""

import os
import sys
from dotenv import load_dotenv

# Import our agents
try:
    from wikipedia_agent import ask_with_wikipedia
    from serpapi_agent import ask_with_search
    from memory_agent import ask_with_memory, show_memory
except ImportError as e:
    print(f"Error importing agents: {e}")
    print("Make sure all agent files are in the same directory")
    sys.exit(1)

def check_environment():
    """Check if required environment variables are set"""
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in .env file")
        return False
    
    if not serpapi_key:
        print("⚠️  SERPAPI_API_KEY not found - SerpAPI agent will not work")
        print("You can still use Wikipedia and Memory agents")
    
    print("✅ Environment configuration looks good!")
    return True

def display_menu():
    """Display the interactive menu"""
    print("\n" + "="*60)
    print("🤖 LangChain Multi-Agent System - Interactive Demo")
    print("="*60)
    print("1. Wikipedia Agent - Ask factual questions")
    print("2. SerpAPI Agent - Search current web information")
    print("3. Memory Agent - Have a conversation with memory")
    print("4. Show Memory Buffer")
    print("5. Test All Agents")
    print("6. Exit")
    print("-"*60)

def test_wikipedia_agent():
    """Test Wikipedia agent with sample questions"""
    print("\n🔍 Testing Wikipedia Agent...")
    questions = [
        "Tell me about artificial intelligence",
        "What is the Eiffel Tower?",
        "Explain quantum computing"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Q: {question}")
        try:
            answer = ask_with_wikipedia(question)
            print(f"   A: {answer[:200]}...")
        except Exception as e:
            print(f"   Error: {e}")

def test_serpapi_agent():
    """Test SerpAPI agent with sample questions"""
    print("\n🌐 Testing SerpAPI Agent...")
    questions = [
        "Current weather in New York",
        "Latest news about AI",
        "Stock price of Tesla today"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Q: {question}")
        try:
            answer = ask_with_search(question)
            print(f"   A: {answer[:200]}...")
        except Exception as e:
            print(f"   Error: {e}")

def test_memory_agent():
    """Test Memory agent with sample conversation"""
    print("\n🧠 Testing Memory Agent...")
    questions = [
        "My name is Alice and I work as a software engineer",
        "What's my name?",
        "What do I do for work?",
        "Can you remember our conversation?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Q: {question}")
        try:
            answer = ask_with_memory(question)
            print(f"   A: {answer}")
        except Exception as e:
            print(f"   Error: {e}")

def interactive_mode(agent_type):
    """Interactive mode for specific agent"""
    print(f"\n🎯 Interactive {agent_type} Mode")
    print("Type 'quit' to return to main menu")
    print("-"*40)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        try:
            if agent_type == "Wikipedia":
                answer = ask_with_wikipedia(question)
            elif agent_type == "SerpAPI":
                answer = ask_with_search(question)
            elif agent_type == "Memory":
                answer = ask_with_memory(question)
            else:
                answer = "Unknown agent type"
            
            print(f"\n🤖 Answer: {answer}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")

def main():
    """Main interactive loop"""
    print("🚀 Starting LangChain Multi-Agent System Demo...")
    
    # Check environment
    if not check_environment():
        return
    
    while True:
        display_menu()
        
        try:
            choice = input("\nSelect an option (1-6): ").strip()
            
            if choice == '1':
                interactive_mode("Wikipedia")
            elif choice == '2':
                interactive_mode("SerpAPI")
            elif choice == '3':
                interactive_mode("Memory")
            elif choice == '4':
                print("\n📚 Current Memory Buffer:")
                show_memory()
            elif choice == '5':
                print("\n🧪 Testing All Agents...")
                test_wikipedia_agent()
                test_serpapi_agent()
                test_memory_agent()
                print("\n✅ All tests completed!")
            elif choice == '6':
                print("\n👋 Thanks for using the LangChain Multi-Agent System!")
                break
            else:
                print("\n❌ Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()