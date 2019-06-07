import torch.nn.functional as F
from torch.autograd import Variable
import lib
import functools
import torch
import logging

logger = logging.getLogger("model")


def clean_sentence(sent, remove_unk=False, remove_eos=True, remove_bos=True):
    if lib.constants.EOS_WORD in sent:
        sent = sent[:sent.index(lib.constants.EOS_WORD) + 1]
    if remove_unk:
        sent = filter(lambda x: x != lib.constants.UNK_WORD, sent)
    if remove_eos:
        if len(sent) > 0 and sent[-1] == lib.constants.EOS_WORD:
            sent = sent[:-1]
    if remove_bos:
        if len(sent) > 0 and sent[0] == lib.constants.BOS_WORD:
            sent = sent[1:]
    return sent


def handle_tags(input_words, pred_words):
    assert len(input_words) == len(pred_words)
    ret = []
    for input_tokens, pred_tokens in zip(input_words, pred_words):
        if lib.constants.URL in pred_tokens or lib.constants.HASH in pred_tokens or lib.constants.MENTION in pred_tokens:
            sent_length = min(len(input_tokens),len(pred_tokens))
            for i in range(sent_length):
                if(pred_tokens[i] == lib.constants.URL or pred_tokens[i] == lib.constants.HASH or pred_tokens[i] == lib.constants.MENTION):
                    pred_tokens[i] = input_tokens[i]
        ret.append(pred_tokens)
    return ret


def handle_numbers(input_words, pred_words):
    assert len(input_words) == len(pred_words)
    ret = []
    for input_tokens, pred_tokens in zip(input_words, pred_words):
        sent_length = min(len(input_tokens),len(pred_tokens))
        for i in range(sent_length):
            if(any(char.isdigit() for char in pred_tokens[i])):
                pred_tokens[i] = input_tokens[i]
        ret.append(pred_tokens)
    return ret


def handle_unk(input, input_words, pred_words, unk_model,unkowns_file=None):
    if(unk_model):
        assert len(input) == len(pred_words)
        ret = []
        for input_tokens, input_words_tokens, pred_tokens in zip(input, input_words, pred_words):
            if lib.constants.UNK_WORD in input_tokens:
                sent_length = min(len(input_tokens),len(pred_tokens))
                for i in range(sent_length):
                    if(input_tokens[i]==lib.constants.UNK_WORD):
                        unk_src = unk_model.encoder.vocab.to_indices(input_words_tokens[i],
                                        eosWord=unk_model.opt.eos,bosWord=unk_model.opt.bos).view(1, -1)
                        #Repeat as many times as the batch size, awful but works
                        unk_src = torch.cat([unk_src]*unk_model.opt.batch_size)
                        unk_src = Variable(unk_src)
                        if input_words_tokens[i] == '' or input_words_tokens[i] == ' ':
                            continue
                        src_lens = Variable(torch.LongTensor([len(p) for p in unk_src]))
                        if unk_model.opt.cuda: unk_src = unk_src.cuda()
                        if unk_model.opt.cuda: src_lens = src_lens.cuda()
                        unk_src = unk_src.t()
                        batch = {}
                        batch['src'] = unk_src, src_lens
                        batch['tgt'] = unk_src, src_lens
                        probs, translation = unk_model.translate(batch)
                        confidence = probs.transpose()[0].max()
                        translation = translation.t().tolist()
                        trsl2wrds = lib.metric.to_words(translation, unk_model.encoder.vocab)
                        if unkowns_file: unkowns_file.writerow([input_words_tokens[i], ''.join(trsl2wrds[0]), confidence])
                        pred_tokens[i] = ''.join(trsl2wrds[0]) if confidence > 50.0 and input_words_tokens[i].isalpha()  else input_words_tokens[i] 
                        if input_words_tokens[i]!=pred_tokens[i]: logger.info('secondary model confidence:{}, unk_word:{}, prediction:{}'.format(confidence, input_words_tokens[i], pred_tokens[i]))
            ret.append(pred_tokens)
    else:
        ret = copy_unks(input, input_words, pred_words)
    return ret


def copy_unks(input, input_words, pred_words):
    assert len(input) == len(pred_words)
    ret = []
    for input_tokens, input_words_tokens, pred_tokens in zip(input, input_words, pred_words):
        if lib.constants.UNK_WORD in input_tokens or lib.constants.UNK_WORD in pred_tokens:
            sent_length = min(len(input_tokens),len(pred_tokens))
            for i in range(sent_length):
                if(input_tokens[i] == lib.constants.UNK_WORD or pred_tokens[i] == lib.constants.UNK_WORD):
                    pred_tokens[i] = input_words_tokens[i]
        ret.append(pred_tokens)
    return ret


def clean_self_toks(inputs, preds, token):
    ret_preds = []
    for input_tokens, pred_tokens in zip(inputs, preds):
        if token in pred_tokens:
            length = min(len(input_tokens), len(pred_tokens))
            for i in range(length):
                if pred_tokens[i] == token:
                    pred_tokens[i] = input_tokens[i]
        ret_preds.append(pred_tokens)
    return ret_preds


def to_words(sents, dict):
    ret = []
    for sent in sents:
        sent = [dict.itos(id) for id in sent]
        sent = clean_sentence(sent, remove_unk=False)
        ret.append(sent)
    return ret


def char_to_words(sents):
    ret = []
    for sent in sents:
        sent = ''.join(sent).split('#')
        ret.append(sent)
    return ret


def compute_single(pair, metric_fn=None):
    input, pred, gold  = pair
    if len(pred) == 0:
        score = 0.
    else:
        score = metric_fn(input, pred, gold)['f1']
    return score


def compute_batch(inputs, preds, golds, metric_fn):
    compute_single_with_metric = functools.partial(compute_single, metric_fn=metric_fn)
    scores = map(compute_single_with_metric, zip(inputs, preds, golds))
    return list(scores)


def compute_numcorrects(dec_logits, targets, pad_masks=None):
    log_dist = F.log_softmax(dec_logits, dim=-1)
    pred_flat = log_dist.max(-1)[1]
    num_corrects = int(pred_flat.eq(targets).masked_select(pad_masks).float().data.sum()) if pad_masks is not None\
        else int(pred_flat.eq(targets).float().data.sum())
    return num_corrects
