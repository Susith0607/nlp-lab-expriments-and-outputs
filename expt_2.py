import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import treebank
from nltk.tag import HiddenMarkovModelTrainer

nltk.download('punkt')
nltk.download('treebank')

train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]

trainer = HiddenMarkovModelTrainer()
hmm_tagger = trainer.train_supervised(train_data)

text = input("Enter a sentence: ")

tokens = word_tokenize(text)
tagged = hmm_tagger.tag(tokens)

print("\nTokens:")
print(tokens)

print("\nPOS Tags (HMM):")
for word, tag in tagged:
    print(f"{word} -> {tag}")

print("\nTag Meanings:")
print("NN  -> Noun")
print("VB  -> Verb")
print("JJ  -> Adjective")
print("RB  -> Adverb")
print("PRP -> Pronoun")
print("DT  -> Determiner")

accuracy = hmm_tagger.accuracy(test_data)

print(f"\nHMM POS Tagger Accuracy: {accuracy * 100:.2f}%")