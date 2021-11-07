from torch import nn
import torch


class EmotionLoss(nn.Module):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self):
        super().__init__()
        self.out_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, output, target):
        output = self.out_softmax(output)
        output = torch.argmax(output, dim=2)
        output = output.to(torch.float)
        target = target.to(torch.float)
        loss = (output - target).mean().sqrt()
        return loss
