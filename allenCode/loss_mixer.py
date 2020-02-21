# from abc import ABC, abstractmethod
from typing import List
from torch import Tensor
from allenCode import losses

class LossMixer:#(ABC)
    epoch=0 # static class variable that can be changed elsewhere

    #@abstractmethod

def mix(logits: List[Tensor], gold: List[Tensor], mask, null_label_id, device):
    cutoff = 15

    if LossMixer.epoch < cutoff:
        return losses.supervised_loss(logits, gold, mask)
    else:
        return losses.reinforce_with_baseline(logits, gold, mask, null_label_id, device)