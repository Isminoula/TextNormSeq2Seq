import torch
import torch.nn.functional as F
from torch.autograd import Variable


def weighted_xent_loss(logits, targets, mask, normalize=True):
    logits_flat = logits.contiguous().view(-1, logits.size(-1))
    targets_flat  = targets.contiguous().view(-1,)
    log_dist = F.log_softmax(logits_flat, dim=-1)
    losses = -log_dist.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
    losses = losses.view(*targets.size())
    losses = losses * mask.float()
    loss = losses.sum()
    loss = loss / mask.float().sum() if normalize else loss
    pred_flat = log_dist.max(1)[1]
    num_corrects = int(pred_flat.eq(targets_flat).masked_select(mask.contiguous().view(-1)).float().data.sum()) \
        if normalize else int(pred_flat.eq(targets_flat).float().data.sum())
    return loss, num_corrects


def sequence_mask(sequence_length,max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda: seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    mask = seq_range_expand < seq_length_expand
    return mask
