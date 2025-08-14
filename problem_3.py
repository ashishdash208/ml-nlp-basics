import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# === Download NLTK data ===
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# === 1. News paragraph ===
# Source: https://www.hindustantimes.com/india-news/supreme-court-stray-dogs-hearing-live-updates-august-14-sc-stray-dogs-order-news-delhi-ncr-noida-shelters-101755135624343.html
text = """
The Supreme Court three-judge bench of Justices Vikram Nath, Sandeep Mehta, and N V Anjaria reserved order in the 
stray dogs Supreme Court case on Thursday, August 14. The hearing follows widespread protests against the top court's 
order directing the blanket removal of Delhi stray dogs from all NCR localities.
"""

# === 2. Tokenization ===
tokens = word_tokenize(text)
print("=== Tokens ===")
print(tokens)

# === 3. Stopword Removal ===
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("\n=== Tokens without Stopwords ===")
print(filtered_tokens)

# === 4. POS Tagging ===
pos_tags = nltk.pos_tag(filtered_tokens)
print("\n=== POS Tagged Tokens ===")
print(pos_tags)

# === 5. Count Nouns, Verbs, Adjectives ===
# POS tag prefixes:
# Nouns: NN, NNS, NNP, NNPS
# Verbs: VB, VBD, VBG, VBN, VBP, VBZ
# Adjectives: JJ, JJR, JJS

counts = Counter()
for word, tag in pos_tags:
    if tag.startswith("NN"):
        counts["Nouns"] += 1
    elif tag.startswith("VB"):
        counts["Verbs"] += 1
    elif tag.startswith("JJ"):
        counts["Adjectives"] += 1

print("\n=== POS Counts ===")
print(counts)
