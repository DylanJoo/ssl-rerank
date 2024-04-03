# naver/splade: The DocumentEncoder is revised from original naver-splade repo.

# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from typing import Optional

import torch
from pyserini.encode import DocumentEncoder, QueryEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer
import sys
import torch.nn as nn
import numpy as np

from collections import defaultdict

def splade_max(tensor, additional_mask=None):
    if additional_mask:
        values, indices = torch.max(
            torch.log(1 + torch.relu(tensor)) * additional_mask.unsqueeze(-1), 
            dim=1)
    else:
        values, indices = torch.max(torch.log(1 + torch.relu(tensor)), dim=1)
    return values

class SpladeDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name_or_dir, tokenizer_name=None, device='cuda:0', pooling='max', minimum=0, quantization_factor=1000):
        self.device = device
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_dir or 'naver/splade-cocondenser-ensembledistil')
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name_or_dir)
        self.has_model = True
        if pooling == 'max':
            self.pooler = splade_max
        else:
            self.pooler = nn.Identity
        # self.l2_norm = l2_norm # seems splade inference will never normalize
        self.quantization_factor = quantization_factor
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()} 
        self.minimum = minimum

    def encode(self, texts=None, titles=None, max_length=256, **kwargs):
        if titles is not None: 
            texts = [f'{title} {text}'.strip() for title, text in zip(titles, texts)]

        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding='longest',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        ).to(self.device) # return_offsets_mapping=False -->  map tokens results

        outputs = self.model(**inputs)
        attention_mask = inputs['attention_mask']

        logits = outputs['logits']
        vector = logits.masked_fill(~attention_mask[..., None].bool(), 0.0)
        vector = self.pooler(vector) # --> bsz vsz

        bow_rep_dict = self._get_bow_rep_dict(vector)
        return bow_rep_dict

    def _get_bow_rep_dict(self, vector):
        cols = torch.nonzero(vector)
        weights = defaultdict(list)
        for col in cols:
            i, j = col.tolist()
            weights[i].append( (j, vector[i, j].cpu().tolist()) )

        def sort_dict(dictionary):
            d = {k: v*self.quantization_factor for (k, v) in dictionary if v >= self.minimum}
            sorted_d = {self.reverse_voc[k]: round(v, 2) for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
            return sorted_d
        return [sort_dict(weight) for i, weight in weights.items()]    


class SpladeQueryEncoder(QueryEncoder):
    def __init__(self, model_name_or_path, tokenizer_name=None, device='cpu'):
        self.device = device
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name_or_path)
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}
        self.weight_range = 5
        self.quant_range = 256

    def encode(self, text, max_length=256, **kwargs):
        inputs = self.tokenizer(
                [text], 
                max_length=max_length, 
                padding='longest',
                truncation=True, 
                add_special_tokens=True, 
                return_tensors='pt'
        ).to(self.device)

        input_ids = inputs['input_ids']
        input_attention = inputs['attention_mask']
        batch_logits = self.model(input_ids)['logits']
        batch_aggregated_logits = splade_max(batch_logits, input_attention)
        # batch_aggregated_logits, _ = torch.max(torch.log(1 + torch.relu(batch_logits)) * input_attention.unsqueeze(-1), dim=1)
        batch_aggregated_logits = batch_aggregated_logits.cpu().detach().numpy()
        raw_weights = self._output_to_weight_dicts(batch_aggregated_logits)
        return self._get_encoded_query_token_wight_dicts(raw_weights)[0]

    def _output_to_weight_dicts(self, batch_aggregated_logits):
        to_return = []
        for aggregated_logits in batch_aggregated_logits:
            col = np.nonzero(aggregated_logits)[0]
            weights = aggregated_logits[col]
            d = {self.reverse_voc[k]: float(v) for k, v in zip(list(col), list(weights))}
            to_return.append(d)
        return to_return

    def _get_encoded_query_token_wight_dicts(self, tok_weights):
        to_return = []
        for _tok_weight in tok_weights:
            _weights = {}
            for token, weight in _tok_weight.items():
                weight_quanted = round(weight / self.weight_range * self.quant_range)
                _weights[token] = weight_quanted
            to_return.append(_weights)
        return to_return

