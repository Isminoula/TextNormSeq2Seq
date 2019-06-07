import json
import logging
import lib
import csv
import os 

logger = logging.getLogger("eval")


class Evaluator(object):
    def __init__(self, model, opt, unk_model=None):
        self.opt = opt
        self.model = model
        self.unk_model = unk_model
        self.criterion = lib.metric.weighted_xent_loss

    def eval(self, data_iter, pred_file=None):
        self.model.eval()
        valid_data = lib.data.Dataset(data_iter, self.opt)
        num_batches = valid_data.num_batches
        val_iter = valid_data.batches()
        all_inputs, all_preds, all_targets, all_others = [], [], [], []
        total_loss = 0
        for i, batch in enumerate(val_iter):
            tgt, tgt_lens = batch['tgt']
            src, src_lens = batch['src']
            tids = batch['tid']
            indices = batch['index']
            outputs = self.model(batch, eval=True)
            mask = lib.metric.sequence_mask(sequence_length=tgt_lens, max_len=tgt.size(0)).transpose(0,1)
            loss, _ = self.model.backward(outputs, tgt, mask, criterion=self.criterion, eval=True, normalize=True)
            probs, predictions = self.model.translate(batch)
            predictions = predictions.t().tolist()
            src, tgt = src.data.t().tolist(), tgt.data.t().tolist()
            src = lib.metric.to_words(src, self.model.encoder.vocab)
            preds = lib.metric.to_words(predictions, self.model.decoder.vocab)
            tgt_sent_words = batch['tgt_sent_words']
            src_sent_words = batch['src_sent_words']

            if(self.opt.input=='char'):
                tgt_sent_words = lib.metric.char_to_words(tgt_sent_words)
                src_sent_words =  lib.metric.char_to_words(src_sent_words)
                preds = lib.metric.char_to_words(preds)
            
            #write predictions from secondary char model to file
            unk_file = csv.writer(open(os.path.join(self.opt.save_dir, "unkowns.csv"), "a"), delimiter='\t') if self.unk_model and not self.opt.interactive else None
            preds = lib.metric.handle_tags(src_sent_words, preds)
            preds = lib.metric.handle_unk(src, src_sent_words, preds, self.unk_model, unk_file)
            if(self.opt.self_tok):
                preds = lib.metric.clean_self_toks(src_sent_words, preds, self.opt.self_tok)
                tgt_sent_words = lib.metric.clean_self_toks(src_sent_words, tgt_sent_words, self.opt.self_tok)

            sent_f1 = [lib.metric.f1([s], [p], [t], spelling=(self.opt.input=='spelling'))['f1'] for s, p, t in zip(src_sent_words, preds, tgt_sent_words)]
            all_inputs.extend(src_sent_words)
            all_preds.extend(preds)
            all_targets.extend(tgt_sent_words)
            all_others.extend([x for x in zip(tids, indices, sent_f1)])
            total_loss += loss

        valid_loss =  total_loss/float(num_batches)
        results = lib.metrics.f1(all_inputs, all_preds, all_targets, spelling=(self.opt.input=='spelling'))
        if self.opt.interactive:
            return all_preds[0]
        else:
            logger.info("correct_norm:{}, total_norm:{}, total_nsw:{}".format(results["correct_norm"], results["total_norm"], results["total_nsw"]))
            logger.info("precision:{}, recall:{}, f1:{}\n".format(results["precision"],results["recall"],results["f1"]))
            self._report(all_inputs[0:5], all_preds[0:5], all_targets[0:5], all_others[0:5])
            if isinstance(pred_file, str):
                self.save_json(all_inputs, all_preds, all_targets, all_others, pred_file)
                logger.info("Corpus F1 score: %.2f" % (results["f1"]*100))
                logger.info("Predictions saved to %s" % pred_file)
            return valid_loss, results['f1']

    def _report(self, inputs, preds, targets, others):
        for input, pred, target, other in zip(inputs, preds, targets, others):
            tid, ind, score = other
            token = '' if self.opt.input == 'spelling' else ' '
            logger.info('ind:{} tid:{} \ninput:{}\ntarget:{}\nprediction:{}\n'.format(
                ind, tid, token.join(input), token.join(target), token.join(pred)))

    def save_json(self, inputs, preds, targets, others, pred_file): #And csv!
        json_entries=[]
        for input, pred, target, other in zip(inputs, preds, targets, others):
            tid, ind, sent_f1 = other
            json_entries.append({"tid":tid,"index":ind,"output":pred,"input":input, "target":target, "score":sent_f1})
        with open(pred_file, "w") as f:
            f.write(json.dumps(json_entries, ensure_ascii=False))
        with open(pred_file+".csv", "w") as f:
            tsvfile = csv.writer(f, delimiter='\t')
            tsvfile.writerow(["ixdex", "score", "output", "input", "target"])
            for x in json_entries:
                tsvfile.writerow([x["index"],x["score"],x["output"],x["input"],x["target"]])