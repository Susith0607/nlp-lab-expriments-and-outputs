import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Input documents
docs = []
n = int(input("Enter number of documents: "))

for i in range(n):
    docs.append(input(f"Enter document {i + 1}: "))

# Input query
query = input("\nEnter search query: ")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
query_vec = vectorizer.transform([query])

# TF-IDF Cosine Similarity
scores = cosine_similarity(query_vec, X)

print("\nTF-IDF Similarity Scores:")
for i, s in enumerate(scores[0]):
    print(f"Document {i + 1}: {round(s, 3)}")

# Latent Semantic Analysis (LSA)
svd = TruncatedSVD(n_components=2, random_state=42)
X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vec)

# LSA Cosine Similarity
lsa_scores = cosine_similarity(query_lsa, X_lsa)

print("\nLSA Similarity Scores:")
for i, s in enumerate(lsa_scores[0]):
    print(f"Document {i + 1}: {round(s, 3)}")

# Most Relevant Document
best = np.argmax(lsa_scores[0])

print("\nMost Relevant Document:")
print(docs[best])