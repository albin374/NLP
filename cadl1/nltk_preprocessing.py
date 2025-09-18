# =====================================
# Text Preprocessing using NLTK
# =====================================

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download resources (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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
    tokens = word_tokenize(doc)
    print(tokens)

# 2. Stopword Removal
stop_words = set(stopwords.words('english'))
print("\n🔹 Stopword Removal:")
for doc in corpus:
    tokens = word_tokenize(doc)
    filtered = [w for w in tokens if w.lower() not in stop_words and w.isalpha()]
    print(filtered)

# 3. Stemming
stemmer = PorterStemmer()
print("\n🔹 Stemming:")
for doc in corpus:
    tokens = word_tokenize(doc)
    stems = [stemmer.stem(w) for w in tokens if w.isalpha()]
    print(stems)

# 4. Lemmatization
lemmatizer = WordNetLemmatizer()
print("\n🔹 Lemmatization:")
for doc in corpus:
    tokens = word_tokenize(doc)
    lemmas = [lemmatizer.lemmatize(w.lower()) for w in tokens if w.isalpha()]
    print(lemmas)
