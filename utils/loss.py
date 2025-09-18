import torch
import torch.nn as nn

class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, outputs, targets):
        # outputs: [batch_size, seq_len, vocab_size]
        # targets: [batch_size, seq_len]
        batch_size, seq_len, vocab_size = outputs.size()
        #Flatten
        outputs = outputs.view(-1, vocab_size)  # [B * seq_len, vocab_size]
        targets = targets.view(-1)              # [B * seq_len]

        loss = self.loss_fn(outputs, targets)
        return loss

# model outputs : (batch_size, seq_len, vocab_size)
# ground truth is (batch_size, seq_len)
#but crossentropy require input: (batch_size, num_classes)
#target: (batch_size,) , so flatten everything