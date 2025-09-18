# llm_assistant.py

from transformers import pipeline

# Load pre-trained models from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
generator = pipeline("text-generation", model="gpt2")
translator = pipeline("translation_en_to_fr", model="t5-base")

# Summarization Example
text = """
Artificial Intelligence is transforming the way industries operate.
From healthcare to finance, AI applications are creating new opportunities,
but also raising ethical concerns about privacy and job replacement.
"""
summary = summarizer(text, max_length=40, min_length=10, do_sample=False)
print("🔹 Summary:", summary[0]['summary_text'])

# Story Generation Example
story = generator("Once upon a time in a futuristic city,", max_length=60, num_return_sequences=1)
print("🔹 Generated Story:", story[0]['generated_text'])

# Translation Example
translation = translator("Machine learning enables computers to learn from data.")
print("🔹 Translation (EN→FR):", translation[0]['translation_text'])
