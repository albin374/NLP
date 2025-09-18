# bow_example.py
from sklearn.feature_extraction.text import CountVectorizer

# Dataset (sample movie reviews / tweets)
docs = [
    "I loved the movie, it was fantastic!",
    "The movie was terrible and boring.",
    "An amazing performance by the lead actor.",
    "I didn’t like the film, it was disappointing."
]

# Create Bag-of-Words model
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(docs)

# Display features (unique words)
print("🔹 Bag of Words - Feature Names")
print(vectorizer.get_feature_names_out())

# Display matrix
print("\n🔹 Bag of Words - Document-Term Matrix")
print(bow_matrix.toarray())
