import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from transformers.modeling_outputs import BaseModelOutput

@dataclass
class CrossEncoderOutput(BaseModelOutput):
    loss: torch.FloatTensor = None
    acc: Optional[Tuple[torch.FloatTensor, ...]] = None
    losses: Optional[Dict[str, torch.FloatTensor]] = None
    q_span: Optional[Tuple[torch.FloatTensor, ...]] = None
    d_span: Optional[Tuple[torch.FloatTensor, ...]] = None

class CrossEncoder(nn.Module):
    """This cross encoder is actually doing same thing with bi-encoders.  """
    def __init__(self, opt, encoder, tokenizer, label_smoothing=False,):
        super(InBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = label_smoothing
        self.tokenizer = tokenizer
        self.encoder = encoder

        self.tau = opt.temperature
        self.tau_span = opt.temperature_span

    def forward(self, tokens, mask, k_tokens, k_mask, stats_prefix="", **kwargs):
        """ In the earlier training, this model indicates the cross-encoder for reranking; 
        thus, the input is a query-context pair 
            tokens: [CLS] <q> [SEP] <d> [SEP] | q_tokens: None

        While in the later, this model will become an encoder of bi-encoders; 
        thus, the input is a seperate query and a context.
            tokens: [CLS] <q> [SEP]           | q_tokens: [CLS] <d> [SEP]
        """

        bsz = len(tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=tokens.device)

        emb = self.encoder(input_ids=tokens, attention_mask=mask, normalize=self.norm_query)

        if (k_tokens is None) and (k_mask is None):
            kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        scores = torch.einsum("id, jd->ij", emb / self.tau, kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        return {'loss': loss, 'acc': accuracy}

    def forward_bidirectional(self, tokens, mask, **kwargs):
        """this forward function has only one list of texts, each can be either a query or a key
        """ 

        bsz = len(tokens)
        labels = torch.arange(bsz, dtype=torch.long, device=tokens.device).view(-1, 2).flip([1]).flatten().contiguous()

        emb = self.encoder(input_ids=tokens, attention_mask=mask, normalize=self.norm_query)

        scores = torch.matmul(emb/self.tau, emb.transpose(0, 1))
        scores.fill_diagonal_(float('-inf'))

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing) 

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        return {'loss': loss, 'acc': accuracy}

    def get_encoder(self):
        return self.encoder

class InBatchForSplade(InBatch):
    """ Reminder: splade's additional parameters """

    def forward(self, tokens, mask, k_tokens, k_mask, **kwargs):
        return super().forward(tokens, mask, k_tokens, k_mask, **kwargs)

class InBatchWithSpan(InBatch):

    def forward(self, tokens, mask, k_tokens, k_mask, **kwargs):

        bsz = len(tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=tokens.device)

        # [objectives] 
        CELoss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        KLLoss = nn.KLDivLoss(reduction='batchmean')
        MSELoss = nn.MSELoss()

        emb, qsemb, qsids = self.encoder(input_ids=tokens, attention_mask=mask, normalize=self.norm_query)
        kemb, ksemb, ksids = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        # [sentence]
        scores = torch.einsum("id, jd->ij", emb / self.tau, kemb)
        loss_0 = CELoss(scores, labels)
        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()

        # [span]
        if self.opt.distil_from_sentence:
            if self.opt.distil_from_sentence.lower() == 'kl':
                # distill from scores
                probs_sents = F.softmax(scores, dim=1)
                scores_spans = torch.einsum("id, jd->ij", qsemb / self.tau_span, ksemb)
                logits_spans = F.log_softmax(scores_spans, dim=1)
                loss_span = KLLoss(logits_spans, probs_sents)
                losses = {'loss_sent': loss_0, 'loss_span': loss_span}

            elif self.opt.distil_from_sentence.lower() == 'mse':
                # distill from embedding (at the normalized vector))
                emb = torch.nn.functional.normalize(emb, dim=-1)
                kemb = torch.nn.functional.normalize(kemb, dim=-1)
                qsemb = torch.nn.functional.normalize(qsemb, dim=-1)
                ksemb = torch.nn.functional.normalize(ksemb, dim=-1)
                loss_span = ( MSELoss(qsemb, emb) + MSELoss(ksemb, kemb) ) / 2
                losses = {'loss_sent': loss_0, 'loss_span': loss_span}

        else:
            ## add loss of (q-span, doc) ## add loss of (query, d-span)
            sscores_1 = torch.einsum("id, jd->ij", qsemb / self.tau_span, kemb)
            loss_1 = CELoss(sscores_1, labels)
            sscores_2 = torch.einsum("id, jd->ij", emb / self.tau_span, ksemb)
            loss_2 = CELoss(sscores_2, labels)
            loss_span = (loss_1 + loss_2) / 2
            losses = {'loss_sent': loss_0, 'loss_span': loss_span}

        loss = loss_0 + loss_span

        return InBatchOutput(
	        loss=loss, 
	        acc=accuracy,
	        losses=losses,
	        q_span=qsids, 
	        d_span=ksids,
	)
