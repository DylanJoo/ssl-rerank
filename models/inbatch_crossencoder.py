import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from transformers.modeling_outputs import BaseModelOutput

@dataclass
class BiCrossEncoderOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    acc: Optional[Tuple[torch.FloatTensor, ...]] = None
    losses: Optional[Dict[str, torch.FloatTensor]] = None

class BiCrossEncoder(nn.Module):
    """This cross encoder is actually doing same thing with bi-encoders.  """
    def __init__(self, opt, encoder, tokenizer, curr_mask_ratio, label_smoothing=False):
        super(BiCrossEncoder, self).__init__()

        self.opt = opt
        self.label_smoothing = label_smoothing
        self.tokenizer = tokenizer
        self.encoder = encoder

        self.tau = opt.temperature
        self.do_biencoder = opt.do_biencoder
        self.curr_mask_ratio = curr_mask_ratio

    def _random_curr_masking(self, x, curr_mask_cands, masked_value=0):
        """
        The replacing(masking) process has two differnt ways: 
            updating the `attention_mask` or updating the `input_ids`.
            This is decided by the masked_value
        """
        for i, cand in enumerate(curr_mask_cands):
            candidates = list(range(cand[0], cand[1]))
            N = math.ceil(self.curr_mask_ratio * len(candidates))
            indices_masked = random.sample(candidates, N)
            x[i, indices_masked] = masked_value # replace 1 into 0 (masked)
        return x


    def forward(self, tokens, attn_masks, curr_mask_cands=None, labels=None, **kwargs):
        """
        In the earlier training, this model indicates the cross-encoder for reranking; 
        thus, the input is a query-context pair: [CLS] <l_tokens> [SEP] <r_tokens> [SEP] 

        While the <r_tokens> can be masksed depending on the training steps.
        """
        bsz = len(tokens)

        # [objectives] 
        CELoss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        ## [curriculum masking] by updating the attention mask
        if self.curr_mask_ratio > 0: # this ratio can be dynamic
            # updating attn_mask 
            # attn_masks = self._random_curr_masking(attn_masks, curr_mask_cands, 0)
            # updating inputs
            tokens = self._random_curr_masking(tokens, curr_mask_cands, self.tokenizer.mask_token_id)

        output = self.encoder(input_ids=tokens, attention_mask=attn_masks)

        # CE: cross-encoder 
        ce_loss = CELoss(output['logits'].view(-1, self.encoder.num_labels), labels.view(-1))
        losses = {'loss_ce': ce_loss}
        predicted_idx = output['logits'].view(-1, self.encoder.num_labels).argmax(-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        # BE: bi-encoder
        if self.do_biencoder:
            be_labels = torch.arange(0, bsz, dtype=torch.long, device=l_tokens.device)
            qemb = output['last_hidden_states'][labels == 1] # this is query plus positive context
            # here need to consider the how many negative should be selected?
            cemb = output['last_hidden_states'][labels == 0] # this is context plus other (negative) context)

            be_scores = torch.einsum("id, jd->ij", qemb / self.tau, cemb)
            be_loss = torch.nn.functional.cross_entropy(be_scores, be_labels, label_smoothing=self.label_smoothing)
            # predicted_idx = torch.argmax(be_scores, dim=-1)
            # accuracy = 100 * (predicted_idx == be_labels).float().mean()
            losses.update({'loss_be': be_loss})

        else:
            be_loss = 0

        loss = ce_loss + be_loss

        return BiCrossEncoderOutput(
                loss=loss, 
                acc=accuracy, 
                losses=losses
        )

    def get_encoder(self):
        return self.encoder

