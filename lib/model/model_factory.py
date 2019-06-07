import lib
import torch
import logging

logger = logging.getLogger("model")


def build_model(vocabs, opt):
    src_vocab, tgt_vocab = vocabs
    encoder = lib.model.EncoderRNN(opt, src_vocab)
    decoder = lib.model.LuongAttnDecoderRNN(opt, tgt_vocab)
    s2smodel = lib.model.Seq2Seq(encoder, decoder, opt)
    optim = create_optim(s2smodel, opt)
    return s2smodel, optim


def create_optim(model, opt):
    trained_params = filter(lambda p: p.requires_grad, model.parameters())
    return lib.train.Optim(trained_params, opt.optim, opt.lr, opt.max_grad_norm,
                           lr_decay=opt.learning_rate_decay, start_decay_after=opt.start_decay_after)


def create_model(vocabs, opt, is_char_model=False):
    model_state = 'model_state_dict'
    optim_state = 'optim_state_dict'
    if opt.load_from is not None or (opt.char_model != None and is_char_model):
        load_loc = opt.load_from if not is_char_model else opt.char_model
        logger.info('Loading model from checkpoint at {}'.format(load_loc))
        if opt.cuda:
            location = lambda storage, loc: storage.cuda(opt.gpu)
        else:
            location = lambda storage, loc: storage
        checkpoint = torch.load(load_loc,map_location=location)
        checkpoint['opt'].cuda = opt.cuda
        model, optim = build_model(vocabs, checkpoint['opt'])
        model.load_state_dict(checkpoint[model_state])
        optim.load_state_dict(checkpoint[optim_state])
        opt.start_epoch = checkpoint['epoch'] + 1
        opt.batch_size = checkpoint['opt'].batch_size
    else:
        logger.info('Building Model')
        model, optim = build_model(vocabs, opt)
    if opt.cuda: model.cuda() # GPU.
    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info('* number of parameters: %d' % nParams)
    return model, optim
