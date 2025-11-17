import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Test if API key is loaded
api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.7, openai_api_key=api_key)

# Create a prompt template
prompt1 = PromptTemplate(
    input_variables=["cuisine"],
    template="Suggest one creative name for a restaurant that serves {cuisine} cuisine. Return only the name."
)

# list of menu items
prompt2 = PromptTemplate(
    input_variables=["restaurant_name"],
    template="Suggest 10 menu items for a restaurant named: {restaurant_name}."
)

# Create sequential chain that preserves restaurant name
chain = (
    RunnablePassthrough.assign(
        restaurant_name=prompt1 | llm | StrOutputParser() | (lambda x: x.strip())
    )
    | RunnablePassthrough.assign(
        menu_items=lambda x: (prompt2 | llm).invoke({"restaurant_name": x["restaurant_name"]})
    )
)

# Run the chain
response = chain.invoke({"cuisine": "bengali"})
print(f"Restaurant Name: {response['restaurant_name']}")
print(f"\nMenu Items:\n{response['menu_items']}")


