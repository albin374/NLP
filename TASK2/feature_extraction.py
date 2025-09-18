# CADL2: Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Dataset 1 (movie reviews / tweets style)
docs1 = [
    "I loved the movie, it was fantastic!",
    "The movie was terrible and boring.",
    "An amazing performance by the lead actor.",
    "I didn’t like the film, it was disappointing."
]

# Dataset 2 (customer product reviews)
docs2 = [
    "The phone has great battery life and an excellent camera.",
    "Terrible sound quality, I will never buy these headphones again.",
    "This laptop is lightweight, fast, and perfect for students.",
    "The smartwatch is overpriced and not very useful."
]

# ---------- Bag of Words ----------
print("🔹 Bag of Words Representation (Movie Reviews)")
vectorizer1 = CountVectorizer()
X_bow1 = vectorizer1.fit_transform(docs1)
print(vectorizer1.get_feature_names_out())
print(X_bow1.toarray())

print("\n🔹 Bag of Words Representation (Product Reviews)")
vectorizer2 = CountVectorizer()
X_bow2 = vectorizer2.fit_transform(docs2)
print(vectorizer2.get_feature_names_out())
print(X_bow2.toarray())

# ---------- TF-IDF ----------
print("\n🔹 TF-IDF Representation (Movie Reviews)")
tfidf1 = TfidfVectorizer()
X_tfidf1 = tfidf1.fit_transform(docs1)
print(tfidf1.get_feature_names_out())
print(X_tfidf1.toarray())

print("\n🔹 TF-IDF Representation (Product Reviews)")
tfidf2 = TfidfVectorizer()
X_tfidf2 = tfidf2.fit_transform(docs2)
print(tfidf2.get_feature_names_out())
print(X_tfidf2.toarray())
