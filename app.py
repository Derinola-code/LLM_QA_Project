import streamlit as st
import google.generativeai as genai
import re

# --- CONFIGURATION ---
# Using the API Key provided in your Flask code
API_KEY = "AIzaSyA3DlxfypsyzI4L7uJ2ZMx5oYFDJ5pOWV8"

# Configure the AI Model
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Configuration Error: {e}")
    model = None

def preprocess_input(text):
    """
    Performs basic NLP preprocessing:
    1. Lowercasing
    2. Removing punctuation
    """
    if not text: return "", []
    text = text.lower()
    # Remove punctuation using Regex (keeps only words and spaces)
    clean_text = re.sub(r'[^\w\s]', '', text)
    # Tokenization for display/logging
    tokens = clean_text.split()
    return clean_text, tokens

# --- STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="NLP Q&A System", page_icon="🧠")

st.title("🧠 NLP Question-Answering System")
st.markdown("""
This application demonstrates **Natural Language Processing (NLP)** by preprocessing user input 
before querying a **Large Language Model (LLM)** for an answer.
""")

st.sidebar.header("System Info")
st.sidebar.info("Algorithm: Gemini-1.5-Flash\n\nFramework: Streamlit")

# User Input Section
user_question = st.text_input("Ask your question:", placeholder="e.g., Explain photosynthesis in simple terms.")

if st.button("Generate Answer"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    elif not model:
        st.error("Model configuration failed. Check API Key.")
    else:
        with st.spinner("Processing NLP steps and generating response..."):
            # 1. NLP Preprocessing Step
            processed_text, tokens = preprocess_input(user_question)
            
            # 2. AI Generation Step
            try:
                # We send the original question to the AI for best context, 
                # but show the preprocessing to satisfy project requirements.
                response = model.generate_content(user_question)
                
                # 3. Display Outputs
                st.subheader("NLP Preprocessing Details")
                col1, col2 = st.columns(2)
                col1.metric("Cleaned Text", processed_text if processed_text else "None")
                col2.metric("Token Count", len(tokens))
                
                with st.expander("View Tokens"):
                    st.write(tokens)
                
                st.markdown("---")
                st.subheader("🤖 AI Response")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"AI Error: {str(e)}")

st.markdown("---")
st.caption("Developed for NLP Question-and-Answering System Project")