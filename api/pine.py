# populate_pinecone.py (Simplified Version)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pinecone import Pinecone
import pickle
from tqdm import tqdm

# --- Configuration ---
# Paste your Pinecone credentials and the Host URL you copied from the website
PINECONE_API_KEY = "pcsk_2wFfLG_U4EXiJ2wboRziKQ8vy8cBgoVdKxkxYidVQAHrS6zUYRbiPhN3Haa16V1MsbZi6V"
INDEX_HOST = "https://chatbot-index-e6zqtzb.svc.aped-4627-b74a.pinecone.io" # <-- Paste the Host URL here

# --- 1. Initialize Connection to Pinecone ---
print("Initializing connection to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
# Connect directly to your index using the host
index = pc.Index(host=INDEX_HOST)
print("Connection successful.")

# --- 2. Load and Prepare Your Data ---
print("Loading data from data.csv...")
data = pd.read_csv('api/data.csv')
data.dropna(subset=['question', 'answer'], inplace=True)
data = data.reset_index(drop=True)
print(f"Loaded {len(data)} rows from CSV.")

# --- 3. Vectorize the Text Data ---
print("Creating text vectors...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['question'])
vector_dimension = X.shape[1]
print(f"Vector dimension is: {vector_dimension}. Make sure your Pinecone index is configured with this dimension.")

# --- 4. Save the Vectorizer for Later Use ---
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Vectorizer has been saved to 'vectorizer.pkl'.")

# --- 5. Upload (Upsert) the Data to Pinecone in Batches ---
print("Uploading vectors to Pinecone...")
batch_size = 100

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(i + batch_size, len(data))
    batch = data.iloc[i:i_end]
    vectors = X[i:i_end].toarray()
    
    to_upsert = []
    for idx, row in enumerate(batch.itertuples()):
        vector_id = str(row.Index)
        vector = vectors[idx].tolist()
        metadata = {"question": row.question, "answer": row.answer}
        to_upsert.append((vector_id, vector, metadata))

    index.upsert(vectors=to_upsert)

print("\nUpload complete!")