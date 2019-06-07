from parameters import parser, change_args
from lib.data.Tweet import Tweet
from lib.data.DataLoader import create_data
import logging
import os
import copy
import lib

logger = logging.getLogger("main")

def train_char_model(args):
    logger.info('*** Character model ***')
    opt = copy.deepcopy(args)
    opt.input = 'spelling'
    train_data, valid_data, test_data, vocab, mappings = lib.data.create_datasets(opt)
    char_model, char_optim = lib.model.create_model((vocab['src'], vocab['tgt']), opt, is_char_model = True)
    char_evaluator = lib.train.Evaluator(char_model, opt)
    char_test_evaluator = lib.train.Evaluator(char_model, opt)
    logger.info(char_model.opt)
    logger.info('Loading test data for character model from "%s"' % opt.testdata)
    logger.info('Loading training data for character model from "%s"' % opt.traindata)
    logger.info(' * Character model vocabulary size. source = %d; target = %d' % (len(vocab['src']), len(vocab['tgt'])))
    logger.info(' * Character model maximum batch size. %d' % opt.batch_size)
    logger.info(char_model)
    if opt.interactive and args.input != 'hybrid':
        while True:
            var = raw_input("Please enter a word to be try spelling model (q to quit): ")
            if var.lower() == 'q': break
            tweets = [Tweet(var.split(), var.split(), '1', '1') for i in range(2)]  # suboptimal but works with minimal changes
            test_data, test_vocab, mappings = create_data(tweets, opt=opt, vocab=vocab, mappings=mappings)
            prediction = char_test_evaluator.eval(test_data)
            print('Prediction is: {}'.format(''.join(prediction)))
    elif opt.eval: # Evaluation only
        logger.info("=======Char eval on test set=============")
        pred_file = os.path.join(opt.save_dir, 'test.pred.char')
        char_test_evaluator.eval(test_data, pred_file=pred_file)
        logger.info("=======Char eval on validation set=============")
        pred_file = os.path.join(opt.save_dir, 'valid.pred.char')
        char_evaluator.eval(valid_data, pred_file=pred_file)
    else: # Training
        char_trainer = lib.train.Trainer(char_model, char_evaluator, train_data, valid_data ,char_optim, opt)
        char_trainer.train(opt.start_epoch, opt.end_epoch)
        logger.info("=======Eval on test set=============")
        pred_file = os.path.join(opt.save_dir, 'test.pred.char')
        char_test_evaluator.eval(test_data, pred_file=pred_file)
        logger.info("=======Eval on validation set=============")
        pred_file = os.path.join(opt.save_dir, 'valid.pred.char')
        char_evaluator.eval(valid_data, pred_file=pred_file)
        logger.info('*** Finished Character model ***\n')
    return char_model


def main():
    opt = parser.parse_args()
    opt = change_args(opt)
    logging.basicConfig(filename=os.path.join(opt.save_dir, 'output.log') if opt.logfolder else None, level=logging.INFO)
    unk_model = train_char_model(opt) if(opt.input in ['hybrid', 'spelling']) else None
    if(opt.input =='spelling'): exit()
    train_data, valid_data, test_data, vocab, mappings = lib.data.create_datasets(opt)
    model, optim = lib.model.create_model((vocab['src'], vocab['tgt']), opt)
    evaluator = lib.train.Evaluator(model, opt, unk_model)
    test_evaluator = lib.train.Evaluator(model, opt, unk_model)
    logger.info(model.opt)
    logger.info('Loading test data from "%s"' % opt.testdata)
    logger.info('Loading training data from "%s"' % opt.traindata)
    logger.info(' * Vocabulary size. source = %d; target = %d' % (len(vocab['src']), len(vocab['tgt'])))
    logger.info(' * Maximum batch size. %d' % opt.batch_size)
    logger.info(model)
    if opt.interactive:
        while True:
            var = raw_input("Please enter the text to be normalized (q to quit): ")
            if var.lower() == 'q': break
            tweets = [Tweet(var.split(), var.split(), '1', '1') for i in range(2)] #suboptimal but works with minimal changes
            test_data, test_vocab, mappings = create_data(tweets, opt=opt, vocab=vocab,mappings=mappings)
            prediction = test_evaluator.eval(test_data)
            print('Prediction is: {}'.format(' '.join(prediction)))
    elif opt.eval: # Evaluation only
        logger.info("=======Eval on test set=============")
        pred_file = os.path.join(opt.save_dir, 'test.pred')
        test_evaluator.eval(test_data, pred_file=pred_file)
        logger.info("=======Eval on validation set=============")
        pred_file = os.path.join(opt.save_dir, 'valid.pred')
        evaluator.eval(valid_data, pred_file=pred_file)
    else: # Training
        trainer = lib.train.Trainer(model, evaluator, train_data, valid_data ,optim, opt)
        trainer.train(opt.start_epoch, opt.end_epoch)
        logger.info("=======Eval on test set=============")
        pred_file = os.path.join(opt.save_dir, 'test.pred')
        test_evaluator.eval(test_data, pred_file=pred_file)
        logger.info("=======Eval on validation set=============")
        pred_file = os.path.join(opt.save_dir, 'valid.pred')
        evaluator.eval(valid_data, pred_file=pred_file)


if __name__ == "__main__":
    main()