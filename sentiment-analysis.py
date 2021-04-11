import nltk
from nltk.corpus import movie_reviews
import random

docs = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

random.shuffle(docs)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
w_features = list(all_words)[:2000]

def document_features(document):
    doc_words = set(document)
    features = {}
    for word in w_features:
        features[word] = (word in doc_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in docs]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
