"""
Word Classifier
Classifies each token into one of:
- ENGLISH
- TANGLISH
- ENTITY
- UNKNOWN
only labels words
"""
import csv
from wordfreq import zipf_frequency as z
class WordClassifier:

    def __init__(self):
        pass

    def is_eng(self,word):
        return z(word.lower(),"en")>2.5

    def classify(self,word):

        if self.is_eng(word):#to check if english 
            return "ENGLISH"
        
        return "UNKNOWN"

if __name__ == "__main__":

    clf= WordClassifier()

    print(clf.classify("happy"))
    print(clf.classify("romba"))
    print(clf.classify("Sindhuja"))