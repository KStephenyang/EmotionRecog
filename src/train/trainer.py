from torch import nn
from transformers import Trainer

from loss.focal_loss import FocalLoss


class EmotionTrainer(Trainer):
    out_bce = nn.BCELoss()
    out_ce = nn.CrossEntropyLoss()
    out_fl = FocalLoss(4)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels") if "labels" in inputs else None
        outputs = model(**inputs)

        loss_love = self.out_fl(outputs['love'], labels[:, 0])
        loss_joy = self.out_fl(outputs['joy'], labels[:, 1])
        loss_fright = self.out_fl(outputs['fright'], labels[:, 2])

        loss_anger = self.out_fl(outputs['anger'], labels[:, 3])
        loss_fear = self.out_fl(outputs['fear'], labels[:, 4])
        loss_sorrow = self.out_fl(outputs['sorrow'], labels[:, 5])

        loss = loss_love + loss_joy + loss_fright + loss_anger + loss_fear + loss_sorrow
        return (loss, outputs) if return_outputs else loss
