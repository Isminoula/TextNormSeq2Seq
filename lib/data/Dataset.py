from torch.autograd import Variable
import torch
import lib

class Dataset(object):
    def __init__(self, data, opt):
        self.DATA_KEYS = data.keys()
        self.TENSOR_KEYS = ['src', 'tgt']
        for key in self.DATA_KEYS:
            setattr(self, key, data[key])
        self.opt = opt
        self.size = len(self.src)
        self.num_batches = (self.size + self.opt.batch_size - 1) // self.opt.batch_size

    def __len__(self):
        return self.num_batches

    def _to_tensor(self, data, return_lens):
        lens = [x.size(0) for x in data]
        max_length = max(lens)
        out = data[0].new(len(data), max_length).fill_(lib.constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            out[i].narrow(0, 0, data_length).copy_(data[i])
        out = out.t_().contiguous()
        if self.opt.cuda: out = out.cuda()
        v = Variable(out)
        lens = Variable(torch.LongTensor(lens))
        if self.opt.cuda: lens = lens.cuda()
        return (v, lens) if return_lens else v

    def batches(self):
        for i in range(self.num_batches):
            s_idx = i * self.opt.batch_size
            e_idx = (i + 1) * self.opt.batch_size
            src_idx_in_data_keys = list(self.DATA_KEYS).index('src')
            value_lists = [getattr(self, key)[s_idx : e_idx] for key in self.DATA_KEYS]
            sorted_value_lists = zip(*sorted(list(zip(*value_lists)),
                                             key=lambda x: -x[src_idx_in_data_keys].size(0)))
            sorted_value_lists = list(sorted_value_lists)
            batch = {}
            for key, value in zip(self.DATA_KEYS, sorted_value_lists):
                batch[key] = value
                if key in self.TENSOR_KEYS:
                    batch[key] = self._to_tensor(value, return_lens=True)
            batch['size'] = len(batch['pos'])
            yield batch
