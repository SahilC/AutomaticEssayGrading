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
        self.fw_count = 0

        #form counts
        self.essay_length = 0
        self.long_word = 0
        self.spelling_errors = 0
        self.sentence_count = 0
        self.avg_sentence_len = 0

        #language model counts
        self.unigrams_count = 0
        self.bigrams_count = 0
        self.trigrams_count = 0

        self.initialize_features(essay)

    def tokenize_sentences(self, essay):
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_detector.tokenize(essay.strip())
        self.sentence_count = len(sents)

        # for i in sents:
        #     self.lexical_diversity(i.lower())
        self.lexical_diversity(essay.lower())

    def word_counts(self,essay):
        word = nltk.word_tokenize(essay.strip())
        self.essay_length = len(word)

        corpus_words = words.words()
        for i in word:
            if i not in corpus_words:
                self.spelling_errors += 1
            if len(i) >= 7:
                self.long_word += 1


        if self.sentence_count != 0:
            self.avg_sentence_len = self.essay_length/self.sentence_count
        else:
            self.avg_sentence_len = 0

        self.pos_counts(word)

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
            elif tag[1].startswith("FW"):
                self.fw_count += 1

    def lexical_diversity(self,sentence):
        sents = " ".join(nltk.word_tokenize(sentence))

        unigrams = [ grams for grams in ngrams(sents.split(), 1)]
        bigrams = [ grams for grams in ngrams(sents.split(), 2)]
        trigram = [ grams for grams in ngrams(sents.split(), 3)]

        self.unigrams_count = len([(item[0], unigrams.count(item)) for item in sorted(set(unigrams))])
        self.bigrams_count = len([(item, bigrams.count(item)) for item in sorted(set(bigrams))])
        self.trigrams_count = len([(item, trigram.count(item)) for item in sorted(set(trigram))])

    def initialize_features(self, essay):
        self.tokenize_sentences(essay)
        self.word_counts(essay)
        #self.lexical_diversity(essay)
