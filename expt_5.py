import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text = input("Enter legal text: ")

tokens = word_tokenize(text)
tags = pos_tag(tokens)

print("\nDetected Named Entities:")

count = 0

for word, tag in tags:
    if tag == "NNP":
        print(word, "-> ENTITY")
        count += 1

actual = int(input("\nEnter actual number of entities: "))

accuracy = (min(count, actual) / max(count, actual)) * 100 if max(count, actual) != 0 else 0

print("\nPredicted Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")