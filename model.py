# coding=utf-8
# email: wangzejunscut@126.com

import torch
from torch import nn
from loss import BinaryLabelSmoothLoss, BinaryFocalLoss

class ElectraForSpellingCheck(nn.Module):
    def __init__(self, pretrained_model):
        super(ElectraForSpellingCheck, self).__init__()
       
        self.hidden_size = pretrained_model.config.hidden_size
        self.pad_token_id = pretrained_model.config.pad_token_id
        
        self.electra = pretrained_model
        self.detector = nn.Linear(self.hidden_size, 1)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        loss_type="bce",
        reduction="mean",
        alpha=None,
        gamma=None,
        label_smoothing=None,
        pos_weight=None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            attention_mask[input_ids==self.pad_token_id] = 0
        
        hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False
        )
        
        sequence_output = hidden_states[0]
        logits = self.detector(sequence_output).squeeze(-1)
        
        loss = None
        if labels is not None:
            if loss_type == "bce":
                if pos_weight is not None:
                    pos_weight = torch.tensor(pos_weight)
                loss_fct = nn.BCEWithLogitsLoss(
                    reduction=reduction,
                    pos_weight=pos_weight
                )
            elif loss_type == "lsr":
                loss_fct = BinaryLabelSmoothLoss(
                    label_smoothing=label_smoothing,
                    reduction=reduction,
                    pos_weight=pos_weight
                )
            elif loss_type == "focal":
                loss_fct = BinaryFocalLoss(
                    alpha=alpha,
                    gamma=gamma,
                    reduction=reduction,
                    pos_weight=pos_weight
                )
            else:
                raise ValueError("Unsupported loss function type!")

            active_loss = attention_mask.view(-1, sequence_output.shape[1]) == 1
            active_logits = logits.view(-1, sequence_output.shape[1])[active_loss]
            active_labels = labels[active_loss]
            loss = loss_fct(active_logits, active_labels.float())

        return (loss, logits) if loss is not None else logits        
