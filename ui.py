import streamlit as st
from dotenv import load_dotenv
import os
from app import chain

st.title("Restaurant Name Suggester")
cuisine = st.sidebar.selectbox("Select Cuisine", ["bengali", "italian", "chinese", "mexican", "indian"], key="cuisine")

def generate_restaurant_name(cuisine):
  response = chain.invoke({"cuisine": cuisine})
  return {
    'resturant_name': response['restaurant_name'],
    'menu_items': response['menu_items']
  }

if cuisine:
  response = generate_restaurant_name(cuisine)
  st.header(response['resturant_name'])
  st.write("Menu Items:")
  for item in response['menu_items'].split(','):
    st.write("- " + item.strip())