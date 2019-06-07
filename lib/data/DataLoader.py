# -*- coding: utf-8 -*-
from .Tweet import Tweet, Preprocessor
from .Dict import Dict
import random
import lib
import json
import copy

class DataLoader(object):
    def __init__(self, tweets, vocab, mappings, opt):
        self.opt = opt
        self.prox_arr = self.get_prox_keys()
        self.mappings = mappings if mappings else {}
        self.tweets, self.source_vocab, self.target_vocab = self.load_data(tweets)
        if(vocab):
            self.source_vocab = vocab["src"]
            self.target_vocab =  vocab["tgt"]
        self.ret = self.encode_tweets()

    def tweets_toIdx(self):
        for tweet in self.tweets:
            input = copy.deepcopy(tweet.input)
            if(self.opt.correct_unique_mappings):
                for index, wi in enumerate(input):
                    mapping = self.mappings.get(wi)
                    if mapping and len(mapping)==1 and list(mapping)[0]!=wi:
                        input[index] = list(mapping)[0]
            tweet.set_inputidx(self.source_vocab.to_indices(input, bosWord=self.opt.bos, eosWord=self.opt.eos))
            tweet.set_outputidx(self.target_vocab.to_indices(tweet.output, bosWord=self.opt.bos, eosWord=self.opt.eos))

    def encode_tweets(self):
        self.tweets_toIdx()
        src_sents, tgt_sents, tgt_sents_words, src_sents_words, indices, tids = [], [], [], [], [], []
        for tweet in self.tweets:
            src_sents.append(tweet.inputidx)
            tgt_sents.append(tweet.outputidx)
            src_sents_words.append(tweet.input)
            tgt_sents_words.append(tweet.output)
            indices.append(tweet.ind)
            tids.append(tweet.tid)

        ret = {'src': src_sents,
               'src_sent_words': src_sents_words,
               'tgt': tgt_sents,
               'tgt_sent_words': tgt_sents_words,
               'pos': range(len(src_sents)),
               'index': indices,
               'tid': tids}

        return ret

    def vector_repr(self, inp_i, inp_o, update_mappings):
        if update_mappings:
            for k in range(len(inp_i)):
                try:
                    self.mappings[inp_i[k].lower()].add(inp_o[k].lower())
                except KeyError:
                    self.mappings[inp_i[k].lower()] =  set()
                    self.mappings[inp_i[k].lower()].add(inp_o[k].lower())

        if(self.opt.self_tok==lib.constants.SELF):
            for i in range(len(inp_i)):
                if(inp_i[i].lower()==inp_o[i].lower()):
                    inp_o[i] =  self.opt.self_tok

        if(self.opt.input=='char'):
            inp_i = list('#'.join(inp_i))
            inp_o = list('#'.join(inp_o))

        return inp_i, inp_o


    def load_data(self, tweets):
        source_vocab = Dict(vocab_size=self.opt.vocab_size, bosWord=self.opt.bos, eosWord=self.opt.eos)
        target_vocab = Dict(vocab_size=self.opt.vocab_size, bosWord=self.opt.bos, eosWord=self.opt.eos)
        if(self.opt.share_vocab):
            target_vocab = source_vocab
        processor = Preprocessor()
        #for test the mappings are predefined and for all other inputs except word level we dont need them, so no updates
        update_mappings = not self.mappings and self.opt.input=='word'
        word_tweets = []
        for tweet in tweets:
            inp_i, pos_i = processor.run(tweet.input,self.opt.lowercase)
            inp_o, pos_o = processor.run(tweet.output, self.opt.lowercase)

            if(self.opt.input == 'spelling'): #character model word2word corrections
                for iword, oword in zip(inp_i, inp_o):
                    if iword and oword and iword.isalnum() and oword.isalnum():
                        if iword == oword and len(iword)>1 and len(oword)>1 and not any(c.isdigit() for c in iword) and  not any(c.isdigit() for c in oword):
                            if random.random() > 0.9 and not self.opt.data_augm:
                                continue
                        iwordv, owordv = self.vector_repr(iword, oword, update_mappings)
                        source_vocab.add_words(iwordv)
                        target_vocab.add_words(owordv)
                        tweet.set_input(iwordv)
                        tweet.set_output(owordv)
                        word_tweets.append(copy.deepcopy(tweet))
                        if(self.opt.data_augm):
                            if random.random() > (1 - self.opt.noise_ratio):
                                if iword == oword and len(iword)>1 and len(oword)>1 and not any(c.isdigit() for c in iword) and  not any(c.isdigit() for c in oword):
                                    iword = self.add_noise(iword)
                                    if(iword == '' or iword == ' '):
                                        continue
                                    iwordv, owordv = self.vector_repr(iword, oword, update_mappings)
                                    source_vocab.add_words(iwordv)
                                    target_vocab.add_words(owordv)
                                    tweet.set_input(iwordv)
                                    tweet.set_output(owordv)
                                    word_tweets.append(tweet)
            else:
                inp_i, inp_o = self.vector_repr(inp_i, inp_o, update_mappings)
                source_vocab.add_words(inp_i)
                target_vocab.add_words(inp_o)
                tweet.set_input(inp_i)
                tweet.set_output(inp_o)
                word_tweets.append(tweet)


        tweets = word_tweets
        if(self.opt.input == 'spelling'):
            same_tw, diff_tw = [], []
            for tweet in tweets:
                if tweet.input == tweet.output:
                    same_tw.append(tweet)
                else:
                    diff_tw.append(tweet)

        source_vocab.makeVocabulary(self.opt.vocab_size)
        source_vocab.makeLabelToIdx()
        target_vocab.makeVocabulary(self.opt.vocab_size)
        target_vocab.makeLabelToIdx()
        if(self.opt.share_vocab):
            assert source_vocab.idx_to_label == target_vocab.idx_to_label
        return tweets, source_vocab, target_vocab


    def add_noise(self, word):
        """
            There are 7 kinds of errors we can introduce for data aug:
            0) forget to "type" a char
            1) swap the placement of two chars
            2) if the word ends in u, y, s, r, extend the last char
            3) if vowel in sentence extend vowel (o, u, e, a, i)
            4-6) misplaced or missing " ' "
            7-10) keyboard errors
        """
        i = random.randint(0,len(word)-1)
        op = random.randint(0, 10)
        if op == 0:
            return word[:i] + word[i+1:]
        if op == 1:
            i += 1
            return word[:i-1] + word[i:i+1] + word[i-1:i] + word[i+1:]
        if op == 2:
            l =word[:-1]
            if l == 'u' or l == 'y' or l == 's' or l == 'r' or l == 'a' or l == 'o' or l == 'i':
                return word + random.randint(1, 5) * l
        if op == 3:
            a = word.find('a')
            e = word.find('e')
            i = word.find('i')
            o = word.find('o')
            u = word.find('u')
            idx = max([a,e,i,o,u])
            if idx != -1:
                return word[:idx] +  random.randint(1, 5) * word[idx] + word[idx:]
        if op == 4:
            idx = word.find("'")
            if idx != -1:
                return word[:idx] + word[idx+1:] + word[idx]
        if op == 5:
            idx = word.find("'")
            if idx != -1:
                return word[:idx-1] + word[idx:idx+1] + word[idx-1:idx] + word[idx+1:]
        if op == 6:
            idx = word.find("'")
            if idx != -1:
                return word[:idx] + word[idx+1:]
        return word[:i] + random.choice(self.prox_arr[word[i]]) + word[i+1:]  #default is keyboard errors
            
    def get_prox_keys(self):
        array_prox = {}
        array_prox['a'] = ['q', 'w', 'z', 'x', 's']
        array_prox['b'] = ['v', 'f', 'g', 'h', 'n', ' ']
        array_prox['c'] = ['x', 's', 'd', 'f', 'v']
        array_prox['d'] = ['x', 's', 'w', 'e', 'r', 'f', 'v', 'c']
        array_prox['e'] = ['w', 's', 'd', 'f', 'r']
        array_prox['f'] = ['c', 'd', 'e', 'r', 't', 'g', 'b', 'v']
        array_prox['g'] = ['r', 'f', 'v', 't', 'b', 'y', 'h', 'n']
        array_prox['h'] = ['b', 'g', 't', 'y', 'u', 'j', 'm', 'n']
        array_prox['i'] = ['u', 'j', 'k', 'l', 'o']
        array_prox['j'] = ['n', 'h', 'y', 'u', 'i', 'k', 'm']
        array_prox['k'] = ['u', 'j', 'm', 'l', 'o']
        array_prox['l'] = ['p', 'o', 'i', 'k', 'm']
        array_prox['m'] = ['n', 'h', 'j', 'k', 'l']
        array_prox['n'] = ['b', 'g', 'h', 'j', 'm']
        array_prox['o'] = ['i', 'k', 'l', 'p']
        array_prox['p'] = ['o', 'l']
        array_prox['q'] = ['w', 'a']
        array_prox['r'] = ['e', 'd', 'f', 'g', 't']
        array_prox['s'] = ['q', 'w', 'e', 'z', 'x', 'c']
        array_prox['t'] = ['r', 'f', 'g', 'h', 'y']
        array_prox['u'] = ['y', 'h', 'j', 'k', 'i']
        array_prox['v'] = ['', 'c', 'd', 'f', 'g', 'b']
        array_prox['w'] = ['q', 'a', 's', 'd', 'e']
        array_prox['x'] = ['z', 'a', 's', 'd', 'c']
        array_prox['y'] = ['t', 'g', 'h', 'j', 'u']
        array_prox['z'] = ['x', 's', 'a']
        array_prox['1'] = ['q', 'w']
        array_prox['2'] = ['q', 'w', 'e']
        array_prox['3'] = ['w', 'e', 'r']
        array_prox['4'] = ['e', 'r', 't']
        array_prox['5'] = ['r', 't', 'y']
        array_prox['6'] = ['t', 'y', 'u']
        array_prox['7'] = ['y', 'u', 'i']
        array_prox['8'] = ['u', 'i', 'o']
        array_prox['9'] = ['i', 'o', 'p']
        array_prox['0'] = ['o', 'p']
        return array_prox



def create_data(data, opt, vocab=None, mappings=None):
    dataload = DataLoader(data, vocab=vocab, mappings=mappings, opt=opt)
    vocab = {}
    vocab['src'] = dataload.source_vocab
    vocab['tgt'] = dataload.target_vocab
    return dataload.ret, vocab, dataload.mappings


def create_datasets(opt):
    train, val = read_file(opt.traindata, opt.valsplit)
    test, _ = read_file(opt.testdata)
    train_data, vocab, mappings = create_data(train, opt=opt)
    if val: val_data, val_vocab, mappings = create_data(val, opt=opt, vocab=vocab, mappings=mappings)
    else: val_data, val_vocab, mappings = train_data, vocab, mappings
    test_data, test_vocab, mappings = create_data(test, opt=opt, vocab=vocab, mappings=mappings)
    return train_data, val_data, test_data, vocab, mappings


def read_file(fn, valsplit=None):
    tweets = []
    with open(fn, 'r') as json_data:
        data = json.load(json_data)
    for tweet in data:
        src_tweet = tweet['input']
        tgt_tweet = tweet['output']
        ind = tweet['index']
        tid = tweet['tid']
        tweets.append(Tweet(src_tweet, tgt_tweet, tid, ind))
    if(valsplit):
        random.shuffle(tweets)
        val = tweets[:valsplit]
        train = tweets[valsplit:]
        return train, val
    return tweets, []

