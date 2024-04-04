# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# /home/dju/datasets/test_collection/bert-base-uncased/corpus.jsonl01.pkl

import os
import glob
import torch
import random
import json
import csv
import numpy as np
import logging
from collections import defaultdict
import torch.distributed as dist

# from src import dist_utils

logger = logging.getLogger(__name__)

def load_dataset(opt, tokenizer):
    """ The original contriever has the multdataset settings.  """
    datasets = {}
    files = glob.glob(os.path.join(opt.train_data_dir, "*.p*"))
    files.sort()
    tensors = []
    if opt.loading_mode == "split":
        files_split = list(np.array_split(files, dist_utils.get_world_size()))[dist_utils.get_rank()]
        for filepath in files_split:
            try:
                tensors.append(torch.load(filepath, map_location="cpu"))
            except:
                logger.warning(f"Unable to load file {filepath}")
    elif opt.loading_mode == "full":
        for fin in files:
            tensors.append(torch.load(fin, map_location="cpu"))
    elif opt.loading_mode == "single":
        tensors.append(torch.load(files[0], map_location="cpu"))

    if len(tensors) == 0:
        return None
    tensor = torch.cat(tensors)
    return Dataset(tensor, opt.chunk_length, tokenizer, opt)


class Dataset(torch.utils.data.Dataset):
    """Monolingual dataset based on a list of paths"""

    def __init__(self, data, chunk_length, tokenizer, opt):

        self.data = data
        self.chunk_length = chunk_length
        # self.max_length = max_length
        self.tokenizer = tokenizer
        self.opt = opt
        self.opt.mask_id = tokenizer.mask_token_id 
        self.generate_offset()

    def __len__(self):
        return (self.data.size(0) - self.offset) // self.chunk_length

    def __getitem__(self, index):
        start_idx = self.offset + index * self.chunk_length
        end_idx = start_idx + self.chunk_length
        tokens = self.data[start_idx:end_idx]
        query = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        context = randomcrop(tokens, self.opt.ratio_min, self.opt.ratio_max)
        return {"query": query, "context": context}

    def generate_offset(self):
        self.offset = random.randint(0, self.chunk_length - 1)


class CollatorForReranking(object):
    def __init__(self, opt, tokenizer, max_length):
        self.opt = opt
        self.tokenizer = tokenizer
        self.max_length = max_length

    def combine_pair(self, left, right):
        # [CLS] <left>  [SEP] <right> [SEP]
        #   0   0 ... 0   0   1111111   0
        token_id = left + [self.tokenizer.bos_token_id] + right
        token_id = token_id[:(self.max_length - 2)]
        token_id = add_bos_eos(token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)
        currmask_id = [0] * (len(left) + 2) + [1] * (len(right) - 1) + [0]
        return token_id, currmask_id

    def __call__(self, batch_examples):

        batch = defaultdict(list)
        batch_size = len(batch_examples)

        ## 1. all negatives: means one example would have 1 positive + (B-1) negatives
        for i, example in enumerate(batch_examples):
            # positive samples # batch['tokens'].append(add_bos_eos(example['context'] + example['query'], self.bos_token_id, None))
            token, currmask = self.combine_pair(example['query'], example['context'])
            batch['pair'].append(token)
            batch['curriculum_mask_candidates'].append(currmask)
            batch['label'].append(1)

            # negative samples from B-1 batch
            other_i = list(range(batch_size))
            other_i.remove(i) # exclude the positive qk pair
            for j in other_i:
                other_example = batch_examples[j]
                token, currmask = self.combine_pair(example['query'], other_example['context'])
                batch['pair'].append(token)
                batch['curriculum_mask_candidates'].append(currmask)
                batch['label'].append(0)

        pair_tokens, pair_masks = build_mask(batch['pair'])
        batch["pair_tokens"] = pair_tokens
        batch["pair_mask"] = pair_mask

        return batch

def randomcrop(x, ratio_min, ratio_max):

    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x) * ratio)
    start = random.randint(0, len(x) - length)
    end = start + length
    crop = x[start:end].clone()
    return crop


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


def apply_augmentation(x, opt, ):
    # if opt.augmentation == "mask":
    #     return torch.tensor(maskword(x, mask_id=opt.mask_id, p=opt.prob_augmentation))
    # elif opt.augmentation == "replace":
    #     return torch.tensor(
    #         replaceword(x, min_random=opt.start_id, max_random=opt.vocab_size - 1, p=opt.prob_augmentation)
    #     )
    # elif opt.augmentation == "delete":
    #     return torch.tensor(deleteword(x, p=opt.prob_augmentation))
    # elif opt.augmentation == "shuffle":
    #     return torch.tensor(shuffleword(x, p=opt.prob_augmentation))
    # else:
    #     if not isinstance(x, torch.Tensor):
    #         x = torch.Tensor(x)
    #     return x

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


# Used for passage retrieval
def load_passages(path):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist")
        return
    logger.info(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        if path.endswith(".jsonl"):
            for k, line in enumerate(fin):
                ex = json.loads(line)
                passages.append(ex)
        else:
            reader = csv.reader(fin, delimiter="\t")
            for k, row in enumerate(reader):
                if not row[0] == "id":
                    ex = {"id": row[0], "title": row[2], "text": row[1]}
                    passages.append(ex)
    return passages
