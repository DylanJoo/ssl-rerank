import os
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
import torch.nn.functional as F

class monoBERT(BertForSequenceClassification):

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        # we dont use the label here
        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=None, 
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )

        # get the score
        logits = model_output['logits'] # the pooled outputs
        hidden_states = model_output['hidden_states'] # the contextualized embeddings

        return {'logits': logits, 'last_hidden_states': hidden_states}
