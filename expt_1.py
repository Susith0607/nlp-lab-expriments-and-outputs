import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

text = input("Enter text: ")

sentences = sent_tokenize(text)
tokens = word_tokenize(text)

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

print("\nSentences:")
print(sentences)

print("\nTokens:")
print(tokens)

print("\nStemmed Words:")
print(stemmed_words)

print("\nLemmatized Words:")
print(lemmatized_words)

print("\nComparison:")
print("Stemming reduces words to root forms, which may not be meaningful.")
print("Lemmatization converts words to meaningful base forms.")