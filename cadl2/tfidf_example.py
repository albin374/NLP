# tfidf_example.py
from sklearn.feature_extraction.text import TfidfVectorizer

# Dataset (same as before)
docs = [
    "I loved the movie, it was fantastic!",
    "The movie was terrible and boring.",
    "An amazing performance by the lead actor.",
    "I didn’t like the film, it was disappointing."
]

# Create TF-IDF model
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs)

# Display features (unique words)
print("🔹 TF-IDF - Feature Names")
print(tfidf.get_feature_names_out())

# Display TF-IDF matrix
print("\n🔹 TF-IDF - Document-Term Matrix")
print(tfidf_matrix.toarray())
