from collections import Counter
from .constants import *
import torch

class Dict(object):
    def __init__(self, vocab_size, bosWord=None, eosWord=None):
        self.vocab = []
        self.vocab_counts = None
        self.vocab_size = vocab_size
        self.bosWord=bosWord
        self.eosWord=eosWord
        self.unkown_words=[]

    @property
    def size(self):
        return len(self.label_to_idx)

    def __len__(self):
        return len(self.label_to_idx)

    def add_words(self, sequence):
        for word in sequence:
            self.vocab.append(word)

    def makeVocabulary(self, vocab_size=None):
        self.vocab = Counter(self.vocab)
        self.vocab_counts = Counter(self.vocab)
        self.vocab = self.prune(vocab_size)
        self.vocab.append(PAD_WORD)
        self.vocab.append(UNK_WORD)
        if(self.bosWord): self.vocab.append(BOS_WORD)
        if(self.eosWord): self.vocab.append(EOS_WORD)


    def makeLabelToIdx(self):
        self.label_to_idx = {PAD_WORD:PAD, UNK_WORD:UNK}
        self.idx_to_label = {PAD:PAD_WORD, UNK:UNK_WORD}
        if(self.bosWord):
            self.bosWord = BOS_WORD
            self.label_to_idx[BOS_WORD]=BOS
            self.idx_to_label[BOS]=BOS_WORD
        if(self.eosWord):
            self.eosWord =  EOS_WORD
            self.label_to_idx[EOS_WORD]=EOS
            self.idx_to_label[EOS]=EOS_WORD
        for item in self.vocab:
            if(item not in self.label_to_idx):
                self.label_to_idx[item] = len(self.label_to_idx)
                self.idx_to_label[len(self.idx_to_label)] =  item
                #TODO: bug when EOS is used and BOS is not used!
                assert item == self.idx_to_label[self.label_to_idx[item]]

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, vocab_size=None):
        if(vocab_size is None): vocab_size = -1
        if vocab_size >=  len(self.vocab) or (vocab_size == -1):
            return sorted(self.vocab, key=self.vocab.get, reverse=True)
        newvocab = self.vocab.most_common(vocab_size)
        self.vocab_counts = self.vocab_counts.most_common(vocab_size)
        # Only keep the `size` most frequent entries.
        return sorted(newvocab, key=newvocab.get, reverse=True)

    def stoi(self, label, default=None):
        try:
            return self.label_to_idx[label]
        except KeyError:
            self.unkown_words.append(label)
            return default

    def itos(self, idx, default=None):
        try:
            return self.idx_to_label[idx]
        except KeyError:
            return default

    def to_indices(self, labels, bosWord=False, eosWord=False):
        vec = []
        if bosWord:
            vec += [self.stoi(BOS_WORD)]
        unk = self.stoi(UNK_WORD)
        vec += [self.stoi(label, default=unk) for label in labels]
        if eosWord:
            vec += [self.stoi(EOS_WORD)]
        return torch.LongTensor(vec)

    def to_labels(self, idx, stop):
        labels = []
        for i in idx:
            labels += [self.itos(i)]
            if i == stop:
                break
        return labels
