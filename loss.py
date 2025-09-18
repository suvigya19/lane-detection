import torch
import torch.nn as nn

class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, outputs, targets):
        # outputs: [batch_size, seq_len, vocab_size]
        # targets: [batch_size, seq_len]
        batch_size, seq_len, vocab_size = outputs.size()

        outputs = outputs.view(-1, vocab_size)  # [B * seq_len, vocab_size]
        targets = targets.view(-1)              # [B * seq_len]

        loss = self.loss_fn(outputs, targets)
        return loss
