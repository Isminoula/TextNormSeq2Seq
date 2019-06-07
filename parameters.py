import torch
from torch.backends import cudnn
from torch import cuda
import numpy as np
import argparse
import random
import os
import logging
import lib

logger = logging.getLogger("main")

parser = argparse.ArgumentParser(description='train.py')
## Data options
parser.add_argument('-traindata', default='dataset/train_data.json', help='Path to train data file')
parser.add_argument('-testdata', default='dataset/test_truth.json',help='Path to the test data file')
parser.add_argument('-valsplit', type=int, default=0,help='Number of examples for validation')
parser.add_argument('-vocab_size', type=int, default=None, help='Limit vocabulary')
parser.add_argument('-lowercase', action='store_true', default=False,help='Converting to lowercase')
parser.add_argument('-share_vocab', action='store_true',default=False,help='Shared vocabulary btw source and target')
parser.add_argument('-eos',action='store_true', default=False,help='Adding EOS token at the end of each sequence')
parser.add_argument('-bos',action='store_true', default=False,help='Adding BOS token in the beginning of each sequence')
parser.add_argument('-self_tok', action='store_true',default=False, help='Special token @self to indicate that the input is to be left alone')
parser.add_argument('-input', default='word', choices=['word', 'char', 'spelling', 'hybrid'],
                    help='character or word level representation, spelling (character model trained on pairs of words) and hybrid (word+spelling)')
parser.add_argument('-maxlen', type=int, default=None,help='Maximum source sequence length')
parser.add_argument('-correct_unique_mappings', action='store_true',default=False, help='Correct unique mappings before training')
parser.add_argument('-char_model', type=str, help='Path to the pretrained char level model')
parser.add_argument('-data_augm', action='store_true',default=False, help='Use data augmentation or not')
## Model options
parser.add_argument('-rnn_type', default='LSTM', choices=['LSTM', 'GRU'], help='Layer type  [LSTM|GRU]')
parser.add_argument('-layers', type=int, default=1,help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-brnn', action='store_true', default=False,help='Use a bidirectional encoder')
parser.add_argument('-rnn_size', type=int, default=300,help='RNN cell hidden size')
parser.add_argument('-emb_size', type=int, default=100,help='Embedding size')
parser.add_argument('-attention', action='store_true', default=False,help='Use attention')
parser.add_argument('-bias', action='store_true', default=False,help='Add bias term')
parser.add_argument('-tie_decoder_embeddings', action='store_true', default=False,
                    help='Share parameters between decoder embeddings and output projection matrix. See https://arxiv.org/abs/1608.05859')
parser.add_argument('-share_embeddings', action='store_true', default=False,
                    help='Share the word embeddings between encoder and decoder. Drastically reduces number of learned parameters.')
parser.add_argument('-dropout', type=float, default=0.2,help='Dropout input of every RNN layer.')
parser.add_argument('-backward_splits', type=int, default=None,help='Backward with smaller batches to save memory.')
parser.add_argument('-teacher_forcing_ratio', type=float, default=0.6,help='Probablity of using teacher forcing (scheduled sampling)')
parser.add_argument('-noise_ratio', type=float, default=0.4,help='% extra noise to add')
## Training
parser.add_argument('-batch_size', type=int, default=32,help='Training batch size')
parser.add_argument('-start_epoch', type=int, default=1,help='Epoch to start training.')
parser.add_argument('-end_epoch', type=int, default=1,help='Number of supervised learning epochs')
parser.add_argument('-optim', default='adam', choices=['sgd', 'adam', 'adagrad', 'adadelta'],help='Optimization method.')
parser.add_argument('-lr', type=float, default=0.01,help='Initial learning rate')
parser.add_argument('-max_grad_norm', type=float, default=5,help='Clip gradients by max global gradient norm. See https://arxiv.org/abs/1211.5063')
parser.add_argument('-learning_rate_decay', type=float, default=0.05,help='Multiply learning with this value after -start_decay_after epochs')
parser.add_argument('-start_decay_after', type=int, default=15,help='Decay learning rate AFTER this epoch')
## GPU
parser.add_argument('-gpu', type=int, default=-1,help='GPU id. Support single GPU only')
parser.add_argument('-log_interval', type=int, default=1,help='Print stats after that many training steps')
parser.add_argument('-save_interval', type=int, default=-1,help='Save model and evaluate after that many training steps')
parser.add_argument('-seed', type=int, default=3435,help='Random seed')
parser.add_argument('-logfolder', action='store_true', default=False, help='Log output to file')
parser.add_argument('-save_dir',default='saving', help='Directory to save model checkpoints')
parser.add_argument('-load_from', type=str, help='Path to a model checkpoint')
## Inference
parser.add_argument('-eval', action='store_true',help='Evaluatation only mode')
parser.add_argument('-interactive', action='store_true',help='Interactive mode')
parser.add_argument('-max_train_decode_len', type=int, default=50,help='Max decoding length during training')
opt = parser.parse_args()

def change_args(opt):
    torch.backends.cudnn.enabled = False
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    if opt.save_dir and not os.path.exists(opt.save_dir): os.makedirs(opt.save_dir)
    logging.basicConfig(filename=os.path.join(opt.save_dir, 'output.log') if opt.logfolder else None, level=logging.INFO)
    if opt.self_tok: opt.self_tok=lib.constants.SELF
    opt.cuda = (opt.gpu != -1) # Set cuda
    if torch.cuda.is_available() and not opt.cuda:
        logger.warning("WARNING: You have a CUDA device, so you should probably run with -gpu 1")
    if opt.cuda: cuda.set_device(opt.gpu)
    if opt.share_embeddings:
        if not opt.share_vocab:
            logger.warning('src/tgt vocab should be the same if you use share_embeddings! Changing share_vocab to True.')
            opt.share_vocab = True
    return opt


