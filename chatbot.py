import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("data.csv")

# Prepare TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['question'])

def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    idx = similarity.argmax()
    score = similarity[0][idx]

    if score > 0.4:  # threshold
        return data.iloc[idx]['answer']
    else:
        return "I'm sorry, I don’t have an answer for that yet."
