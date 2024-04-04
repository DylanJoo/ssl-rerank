import os
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

class monoBERTCrossEncoder:

    def __init__(self, model_name_or_dir, tokenizer_name=None, device='cpu', apply_softmax=False):
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained(model_name_or_dir)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or 'bert-base-uncased')
        self.apply_softmax = apply_softmax

    def predict(self, text_pairs, titles=None, max_length=384):
        """ The `text_pairs` is a list with 
        the first list is query (so all elements in this list should be the same). And
        the second list is documents, which the doc is from top-k retrieved results
        """

        if titles is not None:
            text_pairs[1] = [f'{title} {text}'.strip() for title, text in zip(titles, text_pairs[1])] 

        inputs = self.tokenizer(
                text_pairs[0], text_pairs[1],
                padding=True,
                truncation='longest_first',
                return_tensors='pt',
                max_length=max_length
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs)['logits']

            if self.apply_softmax:
                logits = F.softmax(logits, dim=1)

            scores = logits[:, 1].cpu().detach().numpy()

        return scores

