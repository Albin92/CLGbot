# chatbot.py (Modified)
import pickle
from pinecone import Pinecone

# --- Configuration ---
# Paste your credentials and Host URL here
PINECONE_API_KEY = "pcsk_2wFfLG_U4EXiJ2wboRziKQ8vy8cBgoVdKxkxYidVQAHrS6zUYRbiPhN3Haa16V1MsbZi6V"
INDEX_HOST = "https://chatbot-index-e6zqtzb.svc.aped-4627-b74a.pinecone.io"

# --- Initialization ---
# This code runs only once when the application starts.

# Initialize connection to Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=INDEX_HOST)
    print("Pinecone connection successful.")
except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
    exit()

# Load the saved TfidfVectorizer from the .pkl file
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: vectorizer.pkl not found. Please run the populate_pinecone.py script first.")
    exit()

# --- The Main Chatbot Function ---
def get_response(user_input):
    """
    This function takes user input, vectorizes it, queries Pinecone,
    and returns the best answer.
    """
    if not user_input:
        return "Please ask a question."

    try:
        # 1. Convert the user's question into a vector
        query_vector = vectorizer.transform([user_input]).toarray().tolist()
        
        # 2. Query Pinecone to find the most similar vector
        response = index.query(
            vector=query_vector,
            top_k=1,  # Get the single best match
            include_metadata=True
        )
        print(f"Pinecone response: {response}")
        
        # 3. Process the response
        if response['matches']:
            best_match = response['matches'][0]
            score = best_match['score']
            print(f"Best match score: {score}")
            
            # Adjust this confidence threshold if needed (0.0 to 1.0)
            if score > 0.3: 
                # The answer is in the metadata we stored earlier
                return best_match['metadata']['answer']

        # If no confident match is found
        return "I'm sorry, I don't have an answer for that yet."

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, something went wrong on my end."