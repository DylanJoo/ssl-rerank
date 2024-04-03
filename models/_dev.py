# [todo] add something testing codes for visualization the span
import os
import torch
import transformers
import torch.nn as nn
from transformers import BertModel

"""
Version 1
 - DPR-reader like start and end token extraction
 - Query: segment representation (from CLS), and span representation
        - (a) `boundary_average`: select a start and an end token. And take the average (SBO)
        - (a2) `boundary_project`: select a start and an end token, concatenate them and project to a new embeddings (DensePhrase)
        - (b) `span_extract_average`: extract a span from a start and end token. And take the average (DPR)
        - (c) `span_select_average`: select tokens embeddings from a learned layer, use softmax (UniCoil) [sigmoid is not good]
        - We will possibly need regularization to make this span as short as possible
 - Document: segment representation (from CLS)

Version 2
 - Binary token classification layer
"""

class Contriever(BertModel):
    def __init__(self, config, pooling="mean", span_pooling='span_select_average', **kwargs):
        super().__init__(config, add_pooling_layer=False)
        self.config.pooling = pooling
        self.config.span_pooling = span_pooling

        if 'boundary' in self.config.span_pooling:
            self.outputs = nn.Linear(self.config.hidden_size, 2)
        # elif 'span_extract_average' in self.config.span_pooling:
        #     self.outputs = nn.Linear(self.config.hidden_size, 2)
        elif "span_select_average" in self.config.span_pooling: # may need to add constraints
            self.outputs = nn.Sequential(nn.Linear(self.config.hidden_size, 1), nn.ReLU())
        elif "span_select_sum" in self.config.span_pooling: # may need to add constraints
            self.outputs = nn.Sequential(nn.Linear(self.config.hidden_size, 1), nn.Softmax(1))
        elif "span_select_weird" in self.config.span_pooling: # may need to add constraints
            self.outputs = nn.Sequential(nn.Linear(self.config.hidden_size, 1), nn.Softmax(1))
        else:
            self.outputs = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
        normalize_spans=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_states = model_output["last_hidden_state"]
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        bsz, seq_len = input_ids.size() if input_ids is not None else inputs_embeds.size()[:2]
        emb_size = last_hidden.size(-1)
        kwargs = {
                'hidden': last_hidden, 'mask': attention_mask, 
                'bsz': bsz, 'seq_len': seq_len, 'emb_size': emb_size,
                'normalize': normalize_spans
        }

        # sentence representation
        if 'cls' in self.config.pooling:
            emb = last_hidden[:, 0]
        else:
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)

        # span representation
        if 'boundary_average' in self.config.span_pooling:
            span_emb, span_ids = self._boundary_average(**kwargs)
        elif 'boundary_embedding' in self.config.span_pooling:
            span_emb, span_ids = self._boundary_embedding(**kwargs)
        # elif 'span_extract_average' in self.config.span_pooling:
        #     span_emb, span_ids = self._span_extract_average(**kwargs)
        elif "span_select_average" in self.config.span_pooling: # may need to add constraints
            span_emb, span_ids = self._span_select_average(**kwargs)
        elif "span_select_sum" in self.config.span_pooling: 
            span_emb, span_ids = self._span_select_sum(**kwargs)
        elif "span_select_weird" in self.config.span_pooling: 
            span_emb, span_ids = self._span_select_weird(**kwargs)

        return emb, span_emb, span_ids

    def _boundary_embedding(self, hidden, mask, bsz, seq_len, emb_size, **kwargs): 
        logits = self.outputs(hidden[:, 1:, :]) # exclude CLS 
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.softmax(1).contiguous()
        end_logits = end_logits.softmax(1).contiguous()

        start_id = start_logits.view(bsz, -1).argmax(-1) 
        end_id = end_logits.view(bsz, -1).argmax(-1)
        span_ids = torch.cat([start_id, end_id], dim=-1).view(2, -1)

        # boundary logit: p_i x p_j, shape: bsz seq_len seq_len
        # row-i means the end token, which is at the i-th of seq
        # col-j means the start token, which is at the j-th of seq
        # the trigular matrix has [:, 0] = 1, which mean j=0, i can be [0...N-1]

        # boundary logit: p_i x p_j, shape: bsz seq_len seq_len
        valid_mask = torch.ones(seq_len-1, seq_len-1).triu().T.repeat(bsz,1,1).to(hidden.device)
        boundary_logits = torch.mul(end_logits, start_logits.permute(0, 2, 1))
        boundary_logits = boundary_logits * valid_mask

        # boundary candidates: h'_ij = h_i + h_j, shape: bsz seq_len seq_len embsize
        boundary_embeddings = torch.add( 
                hidden[:, 1:, :].permute(0,2,1)[..., None],    # bsz embsize seqlen 1
                hidden[:, 1:, :].permute(0,2,1)[:, :, None, :] # bsz embsize 1 seqlen
        ).permute(0,2,3,1)  

        span_emb = (boundary_embeddings * boundary_logits[..., None]).view(
                bsz, -1, emb_size
        ).sum(1)
        return span_emb, span_ids

    def _boundary_average(self, hidden, mask, bsz, emb_size, **kwargs): 
        logits = self.outputs(hidden[:, 1:, :]) # exclude CLS 
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.softmax(1).contiguous()
        end_logits = end_logits.softmax(1).contiguous()

        start_id = start_logits.view(bsz, -1).argmax(-1) 
        end_id = end_logits.view(bsz, -1).argmax(-1)
        span_ids = torch.cat([start_id, end_id], dim=-1).view(2, -1)

        failed_span_mask = torch.where(end_id > start_id, 1.0, 0.0)
        span_emb = hidden[:, 1:, :].gather(
                1, span_ids.permute(1, 0)[..., None].repeat(1,1,emb_size)
        ).mean(1)
        span_emb = span_emb * failed_span_mask.view(-1, 1) 
        return span_emb, span_ids

    def _span_extract_average(self, hidden, mask, bsz, seq_len, **kwargs):
        logits = self.outputs(hidden[:, 1:, :]) 
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        start_id = 1 + start_logits.view(bsz, -1).argmax(-1).view(-1, 1)
        end_id = 1 + end_logits.view(bsz, -1).argmax(-1).view(-1, 1)

        span_ids = torch.cat([start_id, end_id], dim=-1)
        ordered = torch.arange(seq_len).repeat(bsz, 1).to(hidden.device)
        extract_span_mask = (start_id <= ordered) & (ordered <= end_id)
        span_emb = torch.mean(hidden * extract_span_mask.unsqueeze(-1), dim=1)
        return span_emb, span_ids

    def _span_select_average(self, hidden, mask, **kwargs):
        select_prob = self.outputs(hidden[:, 1:, :]) # exclude CLS
        span_emb = torch.mean(hidden[:, 1:, :] * select_prob, dim=1) / 256
        top_k_ids = 1 + select_prob.squeeze(-1).topk(10).indices 
        return span_emb, top_k_ids

    def _span_select_sum(self, hidden, mask, normalize, **kwargs):
        select_prob = self.outputs(hidden[:, 1:, :]) # exclude CLS
        # span_emb = torch.sum(hidden[:, 1:, :] * select_prob, dim=1) # sum
        span_emb = (hidden[:, 1:, :] * select_prob).sum(dim=1) / (mask.sum(dim=1) - 1)[..., None]
        top_k_ids = 1 + select_prob.squeeze(-1).topk(10).indices 
        return span_emb, top_k_ids

    def _span_select_weird(self, hidden, mask, **kwargs):
        select_prob = self.outputs(hidden[:, 1:, :]) # exclude CLS
        span_emb = (hidden[:, 1:, :] * select_prob).sum(dim=1) / 256
        top_k_ids = 1 + select_prob.squeeze(-1).topk(10).indices 
        return span_emb, top_k_ids
