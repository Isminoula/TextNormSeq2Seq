import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import lib
import random


class EncoderRNN(nn.Module):
    def __init__(self, opt, vocab):
        super(EncoderRNN, self).__init__()
        self.vocab = vocab
        self.opt = opt
        self.vocab_size = len(self.vocab)
        self.num_directions = 2 if self.opt.brnn else 1
        self.embedding = nn.Embedding(self.vocab_size, opt.emb_size, padding_idx=lib.constants.PAD)
        self.rnn = getattr(nn, self.opt.rnn_type)(
            input_size=self.opt.emb_size,
            hidden_size=opt.rnn_size // self.num_directions,
            num_layers=self.opt.layers,
            dropout=self.opt.dropout,
            bidirectional=self.opt.brnn)

    def forward(self, src, src_lens, hidden=None):
        emb = self.embedding(src)
        packed_emb = nn.utils.rnn.pack_padded_sequence(emb, src_lens)
        packed_outputs, self.hidden = self.rnn(packed_emb, hidden)
        outputs, output_lens =  nn.utils.rnn.pad_packed_sequence(packed_outputs)
        if self.opt.brnn: self.hidden = self._cat_directions(self.hidden)
        return outputs, self.hidden

    def _cat_directions(self, hidden):
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        if isinstance(hidden, tuple): #LSTM
            hidden = tuple([_cat(h) for h in hidden])
        else: #GRU
            hidden = _cat(hidden)
        return hidden


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, opt, vocab):
        super(LuongAttnDecoderRNN, self).__init__()
        self.opt =  opt
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.tanh = nn.Tanh()
        self.embedding =  nn.Embedding(self.vocab_size, opt.emb_size, padding_idx=lib.constants.PAD)
        self.rnn = getattr(nn, self.opt.rnn_type)(
            input_size=self.opt.emb_size,
            hidden_size=self.opt.rnn_size,
            num_layers=self.opt.layers,
            dropout=self.opt.dropout)

        if self.opt.attention:
            self.W_a = nn.Linear(self.opt.rnn_size, self.opt.rnn_size, bias=opt.bias)
            self.W_c = nn.Linear(self.opt.rnn_size + self.opt.rnn_size, self.opt.rnn_size, bias=opt.bias)

        if self.opt.tie_decoder_embeddings and self.vocab_size!=1:
            self.W_proj = nn.Linear(self.opt.rnn_size, self.opt.emb_size, bias=opt.bias)
            self.W_s = nn.Linear(self.opt.emb_size, self.vocab_size, bias=opt.bias)
            self.W_s.weight = self.embedding.weight
        else:
            self.W_s = nn.Linear(self.opt.rnn_size, self.vocab_size, bias=opt.bias)

    def forward(self, src, src_lens, encoder_outputs, decoder_hidden):
        emb = self.embedding(src.unsqueeze(0))
        decoder_output, self.decoder_hidden = self.rnn(emb, decoder_hidden)
        decoder_output = decoder_output.transpose(0,1)
        if self.opt.attention:
            attention_scores = torch.bmm(decoder_output, self.W_a(encoder_outputs).transpose(0,1).transpose(1,2))
            attention_mask = lib.metric.sequence_mask(src_lens).unsqueeze(1)
            attention_scores.data.masked_fill_(1 - attention_mask.data, -float('inf'))
            attention_weights = F.softmax(attention_scores.squeeze(1), dim=1).unsqueeze(1)
            context_vector = torch.bmm(attention_weights, encoder_outputs.transpose(0,1))
            concat_input = torch.cat([context_vector, decoder_output], -1)
            concat_output = self.tanh(self.W_c(concat_input))
            attention_weights = attention_weights.squeeze(1)
        else:
            attention_weights = None
            concat_output = decoder_output
        if self.opt.tie_decoder_embeddings and self.vocab_size!=1:
            output = self.W_s(self.W_proj(concat_output))
        else:
            output = self.W_s(concat_output)
        output = output.squeeze(1)
        del src_lens
        return output, self.decoder_hidden, attention_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, opt):
        super(Seq2Seq, self).__init__()
        self.torch = torch.cuda if opt.cuda else torch
        self.encoder = encoder
        self.decoder = decoder
        self.opt = opt
        if opt.share_embeddings:
            self.encoder.embedding.weight = self.decoder.embedding.weight

    def forward(self, batch, eval=False):
        tgt, tgt_lens = batch['tgt']
        src, src_lens = batch['src']
        batch_size = src.size(1)
        assert(batch_size == tgt.size(1))
        input_seq = Variable(torch.LongTensor([lib.constants.BOS] * batch_size))
        decoder_outputs = Variable(torch.zeros(self.opt.max_train_decode_len, batch_size, self.decoder.vocab_size))
        if self.opt.cuda: input_seq, decoder_outputs = input_seq.cuda(), decoder_outputs.cuda()
        max_tgt_len = tgt.size()[0]
        encoder_outputs, encoder_hidden = self.encoder(src, src_lens.data.tolist())
        decoder_hidden = encoder_hidden
        use_teacher_forcing = False if eval else random.random() < self.opt.teacher_forcing_ratio
        for t in range(max_tgt_len):
            decoder_output, decoder_hidden, attention_weights = self.decoder(input_seq, src_lens, encoder_outputs, decoder_hidden)
            decoder_outputs[t] = decoder_output
            if use_teacher_forcing:
                input_seq = tgt[t]
            else:
                topv, topi = decoder_output.topk(1)
                input_seq = topi.squeeze()
        return decoder_outputs

    def backward(self, outputs, tgt_seqs, mask, criterion, eval=False, normalize=True):
        max_tgt_len = tgt_seqs.size()[0]
        logits = outputs[:max_tgt_len]
        loss, num_corrects = criterion(logits, tgt_seqs, mask, normalize=normalize)
        if(not eval): loss.backward()
        return loss.item(), num_corrects

    def translate(self, batch):
        tgt, tgt_lens = batch['tgt']
        src, src_lens = batch['src']
        batch_size = src.size(1)
        assert (batch_size == tgt.size(1))
        input_seq = Variable(torch.LongTensor([lib.constants.BOS] * batch_size))
        decoder_outputs = Variable(torch.zeros(self.opt.max_train_decode_len, batch_size, self.decoder.vocab_size))
        if self.opt.cuda: input_seq, decoder_outputs = input_seq.cuda(), decoder_outputs.cuda()
        encoder_outputs, encoder_hidden = self.encoder(src, src_lens.data.tolist())
        decoder_hidden = encoder_hidden
        if self.opt.attention: all_attention_weights = torch.zeros(self.opt.max_train_decode_len, src.size(1), len(src))
        end_of_batch_pred = np.array([lib.constants.EOS] * len(src_lens))
        preds = np.ones((self.opt.max_train_decode_len, len(src_lens))) * 2
        probs = np.ones((self.opt.max_train_decode_len, len(src_lens))) * 2
        for t in range(self.opt.max_train_decode_len):
            decoder_output, decoder_hidden, attention_weights = self.decoder(input_seq, src_lens, encoder_outputs, decoder_hidden)
            if self.opt.attention:
                all_attention_weights[t] = attention_weights.cpu().data
            prob, token_ids = decoder_output.data.topk(1)
            token_ids = token_ids.squeeze()
            prob = prob.squeeze()
            preds[t,:] = token_ids
            probs[t,:] = prob
            input_seq = Variable(token_ids)
            if np.sum(np.equal(token_ids.cpu().numpy(),end_of_batch_pred)) == len(src):
                break
        preds = torch.LongTensor(preds)
        return probs, preds