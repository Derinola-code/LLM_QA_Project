import google.generativeai as genai
import re

API_KEY = "AIzaSyA3DlxfypsyzI4L7uJ2ZMx5oYFDJ5pOWV8"

def preprocess_input(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return text, tokens

def main():
    print("="*50)
    print("NLP Q&A SYSTEM - CLI MODE")
    print("="*50)
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    while True:
        user_input = input("\nAsk a question (type 'exit' to quit): ")
        if user_input.lower() == 'exit': break
        
        # NLP Step
        processed, tokens = preprocess_input(user_input)
        print(f"[NLP Log] Tokens: {tokens}")

        # AI Step
        try:
            response = model.generate_content(user_input)
            print(f"\nAI Answer:\n{response.text}\n" + "="*50)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()