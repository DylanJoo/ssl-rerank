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

import faiss
import torch
from pyserini.encode import DocumentEncoder
from pyserini.search.faiss import QueryEncoder
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
import sys

class ContrieverDocumentEncoder(DocumentEncoder):
    def __init__(self, model_name_or_dir, tokenizer_name=None, device='cuda:0', pooling='mean', l2_norm=False):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name_or_dir or 'facebook/contriever', add_pooling_layer=False)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or 'facebook/contriever')
        self.has_model = True
        self.pooling = pooling
        self.l2_norm = l2_norm

    def encode(self, texts=None, titles=None, max_length=256, **kwargs):
        """
        As we would like to generalize encoder to all the other domains, 
        we did not use the [SEP] token to seperate the title and content (texts).
        """
        if titles is not None: 
            texts = [f'{title} {text}'.strip() for title, text in zip(titles, texts)]

        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding='longest',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        ).to(self.device) 
        outputs = self.model(**inputs)

        if self.pooling == "mean":
            embeddings = self._mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu().numpy()
        elif self.pooling == 'cls':
            embeddings = outputs[0][:, 0, :].detach().cpu().numpy()
        if self.l2_norm:
            embeddings = normalize(embeddings, axis=1, norm='l2')
        return embeddings

class ContrieverQueryEncoder(QueryEncoder):
    def __init__(self, model_name_or_dir, tokenizer_name=None, device='cpu', pooling='mean', l2_norm=False):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name_or_dir or 'facebook/contriever', add_pooling_layer=False)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or 'facebook/contriever')
        self.pooling = pooling
        self.l2_norm = l2_norm

    def encode(self, query: str, **kwargs):
        inputs = self.tokenizer(
            [query],
            max_length=64,
            padding='longest',
            truncation='only_first',
            add_special_tokens=True,
            return_tensors='pt'
        )
        inputs.to(self.device)
        outputs = self.model(**inputs)

        if self.pooling == "mean":
            embeddings = self._mean_pooling(outputs, inputs['attention_mask']).detach().cpu().numpy()
        elif self.pooling == 'cls':
            embeddings = outputs[0][:, 0, :].detach().cpu().numpy()
        if self.l2_norm:
            faiss.normalize_L2(embeddings)
        return embeddings.flatten()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
