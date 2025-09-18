# ner_example.py

import spacy
import pandas as pd

# Load spaCy's pre-trained English model
nlp = spacy.load("en_core_web_sm")

# Sample dataset (job postings / scientific articles)
texts = [
    "John Doe is a software engineer at Google.",
    "Jane Smith works for Microsoft as a data scientist.",
    "Dr. Alan Turing was a famous computer scientist at Princeton University."
]

# Create lists to store extracted entities
persons = []
organizations = []

# Process each text
for doc in texts:
    spacy_doc = nlp(doc)
    for ent in spacy_doc.ents:
        if ent.label_ == "PERSON":
            persons.append(ent.text)
        if ent.label_ == "ORG":
            organizations.append(ent.text)

# Combine into a structured table
df = pd.DataFrame({
    "Person": persons,
    "Organization": organizations
})

print("🔹 Extracted Named Entities (Structured Table):")
print(df)
