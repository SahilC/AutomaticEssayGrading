from nltk.corpus import words
from nltk import ngrams
import nltk

class Features:
    def __init__(self,essay):
        self.google_snippets_match = 0

        #POS counts
        self.noun_count = 0
        self.adj_count = 0
        self.verb_count = 0
        self.adv_count = 0

        #form counts
        self.essay_length = 0
        self.long_word = 0
        self.spelling_errors = 0
        self.sentence_count = 0
        self.avg_sentence_len = 0

        #language model counts
        self.unigram_count = 0
        self.bigram_count = 0
        self.trigram_count = 0

        self.initialize_features(essay)

    def tokenize_sentences(self, essay):
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_detector.tokenize(essay.strip())
        self.sentence_count = len(sents)

    def word_counts(self,essay):
        words = nltk.word_tokenize(essay.strip())
        self.essay_length = len(words)

        corpus_words = words.words()
        for i in words:
            if i not in corpus_words:
                self.spelling_errors += 1
            if len(i) >= 7:
                self.long_word += 1


        if self.sentence_count != 0:
            self.avg_sentence_len = self.essay_length/self.sentence_count
        else:
            self.avg_sentence_len = 0

        self.pos_counts(self,words)

    def pos_counts(self,tokens):
        tags = nltk.pos_tag(tokens)
        for tag in tags:
            if tag[1].startswith("NN"):
                self.noun_count += 1
            elif tag[1].startswith("JJ"):
                self.adj_count += 1
            elif tag[1].startswith("RB"):
                self.adv_count += 1
            elif tag[1].startswith("VB"):
                self.adv_count += 1

    def lexical_diversity(self,essay):
        pass

    def initialize_features(self, essay):
        self.tokenize_sentences(essay)
        self.word_counts(essay)
        self.lexical_diversity(essay)


if __name__ == "__main__":
    f = Features("This is me. This is my life.  Woah, woah woahooooaaa.")
    print f.sentence_count
    print f.essay_length
    print f.long_word
    print f.avg_sentence_len
