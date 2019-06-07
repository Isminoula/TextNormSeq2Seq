import os
import time
import torch
import logging
import lib

logger = logging.getLogger("train")


class Trainer(object):
    def __init__(self, model, evaluator, train_data, eval_data, optim, opt, test_eval=None):
        self.model = model
        self.evaluator = evaluator
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.opt = opt
        self.test_eval = test_eval
        self.criterion = lib.metric.weighted_xent_loss

    def train(self, start_epoch, end_epoch):
        if(self.opt.save_interval==-1): self.opt.save_interval=end_epoch+1
        for epoch in range(start_epoch, end_epoch + 1):
            logger.info('\n* TextNorm epoch *')
            logger.info('Model optim lr: %g' % self.optim.lr)
            total_loss, total_accuracy = self.train_epoch(epoch)
            logger.info('Train loss: %.2f' % total_loss)
            logger.info('Train total_accuracy: %.2f' % total_accuracy)
            valid_loss, valid_f1 = self.evaluator.eval(self.eval_data)
            self.optim.update_lr(valid_loss, epoch)
            if epoch % self.opt.save_interval == 0 or epoch==end_epoch:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'opt': self.opt,
                    'epoch': epoch,
                }
                model_name = os.path.join(self.opt.save_dir, "model_%d" % epoch)
                model_name += "_"+self.opt.input+".pt"
                torch.save(checkpoint, model_name)
                logger.info('Save model as %s' % model_name)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_time = time.time()
        train_data = lib.data.Dataset(self.train_data, self.opt)
        num_batches = train_data.num_batches
        train_iter = train_data.batches()
        total_loss, total_corrects, total_tgts = 0, 0, 0
        for i, batch in enumerate(train_iter):
            self.model.train()
            tgt, tgt_lens = batch['tgt']
            src, src_lens = batch['src']
            outputs = self.model(batch)
            self.model.zero_grad()
            pad_masks = lib.metric.sequence_mask(sequence_length=tgt_lens, max_len=tgt.size(0)).transpose(0,1)
            loss, num_corrects = self.model.backward(outputs, tgt, pad_masks, criterion=self.criterion)
            num_words = (tgt.data.ne(lib.constants.PAD).sum() + src.data.ne(lib.constants.PAD).sum()).item()
            num_tgts = tgt_lens.data.sum().item()
            total_loss += loss
            total_corrects += num_corrects
            total_tgts += num_tgts
            self.optim.step()
            if (i + 1) % self.opt.log_interval == 0:
                words_pers = int(num_words / (time.time() - epoch_time))
                accuracy = 100 * (num_corrects/float(num_tgts))
                logger.info('Epoch %3d,  %6d/%d batches  loss:%f,  num_words:%d,  accuracy:%f' %
                      (epoch, i + 1, num_batches, loss, words_pers, accuracy))
        return total_loss/float(num_batches), 100*(total_corrects/float(total_tgts))