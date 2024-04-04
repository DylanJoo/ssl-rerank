import os
import torch
import random
import itertools
import numpy as np
import logging
import glob
from collections import defaultdict
from transformers import DataCollatorForWholeWordMask


logger = logging.getLogger(__name__)

def load_dataset(opt, tokenizer):
    files = glob.glob(os.path.join(opt.train_data_dir, "*")) 
    return Dataset(files[0], opt.chunk_length, tokenizer, opt)

class Dataset(torch.utils.data.Dataset):
    """The dataset for ICT is based on random cropping.
    However, we select a span as query, and leave the other as the context (the document)
    Some details are based on the Megatron-LM implementation (see here: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/ict_dataset.py)

    The option's parameters are:
        - augmentation: 'none'
    """

    def __init__(self, path, chunk_length, tokenizer, opt):
        # [todo] is this the most handy way to load dataset? inherit?
        from datasets import load_dataset as load
        self.data = load('json', 
            data_files=path, 
            ignore_verifications=True,
            keep_in_memory=True)['train']
        self.chunk_length = chunk_length
        self.tokenizer = tokenizer
        self.query_in_block_prob = opt.query_in_block_prob
        self.use_one_sent_docs = False
        self.binary_head = False
        self.rng = random.Random(42)
        self.opt = opt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        spans_in_doc = self.data[index]['spans']
        rand_sent_idx = self.rng.randint(0, len(spans_in_doc) - 1) 
        # [todo] maybe it's better to select the top and bottom?

        if self.rng.random() < self.query_in_block_prob:
            query = spans_in_doc[rand_sent_idx].copy() # presever it 10%
        else:
            query = spans_in_doc.pop(rand_sent_idx) # discard it 90%

        query = self.truncate(query)
        query = add_bos_eos(query, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)

        context = list(itertools.chain(*spans_in_doc))
        context = apply_augmentation(context, self.opt)
        context = self.truncate(context)
        context = add_bos_eos(context, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)

        return {"query": query, "context": context}

    def truncate(self, example):
        tgt_len = self.chunk_length - self.tokenizer.num_special_tokens_to_add(False)
        if len(example) <= tgt_len:
            return example
        trunc = len(example) - tgt_len
        trunc_left = random.randint(0, trunc)
        trunc_right = trunc - trunc_left

        truncated = example[trunc_left:]
        if trunc_right > 0:
            truncated = truncated[:-trunc_right]

        if not len(truncated) == tgt_len:
            print(len(example), len(truncated), trunc_left, trunc_right, tgt_len, flush=True)
            raise ValueError
        return truncated


class Collator(DataCollatorForWholeWordMask):

    def __init__(self, opt, tokenizer, **kwargs):
        self.opt = opt
        self.do_mlm = False
        super().__init__(tokenizer, **kwargs)

    def __call__(self, batch_examples):

        batch = defaultdict(list)
        for example in batch_examples:
            for k, v in example.items():
                batch[k].append(v)

        q_tokens, q_mask = build_mask(batch["query"])
        ctx_tokens, ctx_mask = build_mask(batch["context"])

        batch["q_tokens"] = q_tokens
        batch["q_mask"] = q_mask
        batch["k_tokens"] = ctx_tokens
        batch["k_mask"] = ctx_mask

        return batch


def build_mask(tensors):
    shapes = [x.shape for x in tensors]
    maxlength = max([len(x) for x in tensors])
    returnmasks = []
    ids = []
    for k, x in enumerate(tensors):
        returnmasks.append(torch.tensor([1] * len(x) + [0] * (maxlength - len(x))))
        ids.append(torch.cat((x, torch.tensor([0] * (maxlength - len(x))))))
    ids = torch.stack(ids, dim=0).long()
    returnmasks = torch.stack(returnmasks, dim=0).bool()
    return ids, returnmasks

def add_token(x, token):
    x = torch.cat((torch.tensor([token]), x))
    return x


def deleteword(x, p=0.1):
    mask = np.random.rand(len(x))
    x = [e for e, m in zip(x, mask) if m > p]
    return x


def replaceword(x, min_random, max_random, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else random.randint(min_random, max_random) for e, m in zip(x, mask)]
    return x


def maskword(x, mask_id, p=0.1):
    mask = np.random.rand(len(x))
    x = [e if m > p else mask_id for e, m in zip(x, mask)]
    return x


def shuffleword(x, p=0.1):
    count = (np.random.rand(len(x)) < p).sum()
    """Shuffles any n number of values in a list"""
    indices_to_shuffle = random.sample(range(len(x)), k=count)
    to_shuffle = [x[i] for i in indices_to_shuffle]
    random.shuffle(to_shuffle)
    for index, value in enumerate(to_shuffle):
        old_index = indices_to_shuffle[index]
        x[old_index] = value
    return x


def apply_augmentation(x, opt):
    if opt.augmentation == "mask":
        return torch.tensor(maskword(x, mask_id=opt.mask_id, p=opt.prob_augmentation))
    elif opt.augmentation == "replace":
        return torch.tensor(
            replaceword(x, min_random=opt.start_id, max_random=opt.vocab_size - 1, p=opt.prob_augmentation)
        )
    elif opt.augmentation == "delete":
        return torch.tensor(deleteword(x, p=opt.prob_augmentation))
    elif opt.augmentation == "shuffle":
        return torch.tensor(shuffleword(x, p=opt.prob_augmentation))
    else:
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        return x


def add_bos_eos(x, bos_token_id, eos_token_id):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    if bos_token_id is None and eos_token_id is not None:
        x = torch.cat([x.clone().detach(), torch.tensor([eos_token_id])])
    elif bos_token_id is not None and eos_token_id is None:
        x = torch.cat([torch.tensor([bos_token_id]), x.clone().detach()])
    elif bos_token_id is None and eos_token_id is None:
        pass
    else:
        x = torch.cat([torch.tensor([bos_token_id]), x.clone().detach(), torch.tensor([eos_token_id])])
    return x
