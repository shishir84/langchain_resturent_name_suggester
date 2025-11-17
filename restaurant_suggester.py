import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    temperature=0.7,  # Higher temperature for creative names
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create a prompt template
prompt_template = ChatPromptTemplate.from_template(
    """You are a creative restaurant naming expert. 
    Generate 5 unique and catchy restaurant names for a {cuisine_type} restaurant.
    The restaurant should have a {atmosphere} atmosphere.
    
    Provide names that are:
    - Memorable and easy to pronounce
    - Relevant to the cuisine type
    - Suitable for the atmosphere
    
    Cuisine: {cuisine_type}
    Atmosphere: {atmosphere}
    
    Restaurant Names:"""
)

def suggest_restaurant_names(cuisine_type, atmosphere):
    """Generate restaurant name suggestions"""
    # Create the chain
    chain = prompt_template | llm
    
    # Execute the chain
    response = chain.invoke({
        "cuisine_type": cuisine_type,
        "atmosphere": atmosphere
    })
    
    return response.content

# Example usage
if __name__ == "__main__":
    print("🍽️ Restaurant Name Suggester")
    print("="*40)
    
    # Test different combinations
    examples = [
        ("Italian", "romantic"),
        ("Japanese", "modern"),
        ("Mexican", "casual"),
        ("French", "upscale"),
        ("Indian", "family-friendly"),
        ("Thai", "trendy")
    ]
    
    for cuisine, atmosphere in examples:
        print(f"\n{cuisine} - {atmosphere}:")
        print("-" * 30)
        names = suggest_restaurant_names(cuisine, atmosphere)
        print(names)
        print()
    
    # Interactive mode
    print("\n" + "="*40)
    print("Interactive Mode - Enter your preferences:")
    
    while True:
        try:
            cuisine = input("\nEnter cuisine type (or 'quit' to exit): ").strip()
            if cuisine.lower() == 'quit':
                break
            
            atmosphere = input("Enter atmosphere: ").strip()
            
            if cuisine and atmosphere:
                print(f"\n🎯 Generating names for {cuisine} restaurant with {atmosphere} atmosphere...")
                names = suggest_restaurant_names(cuisine, atmosphere)
                print(names)
            else:
                print("Please enter both cuisine type and atmosphere.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")