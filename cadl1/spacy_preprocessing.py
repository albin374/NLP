# =====================================
# Text Preprocessing using spaCy
# =====================================

import spacy

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Sample Text Corpus
corpus = [
    "The new AI model is revolutionizing natural language processing.",
    "Students are studying hard for their upcoming exams.",
    "The football match was exciting and thrilling to watch!"
]

print("\n🔹 Original Corpus:")
for doc in corpus:
    print("-", doc)

# 1. Tokenization
print("\n🔹 Tokenization:")
for doc in corpus:
    spacy_doc = nlp(doc)
    print([token.text for token in spacy_doc])

# 2. Stopword Removal
print("\n🔹 Stopword Removal:")
for doc in corpus:
    spacy_doc = nlp(doc)
    filtered = [token.text for token in spacy_doc if not token.is_stop and token.is_alpha]
    print(filtered)

# 3. Lemmatization (spaCy handles it better than NLTK)
print("\n🔹 Lemmatization:")
for doc in corpus:
    spacy_doc = nlp(doc)
    lemmas = [token.lemma_ for token in spacy_doc if token.is_alpha]
    print(lemmas)
