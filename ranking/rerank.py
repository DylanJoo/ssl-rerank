from operator import itemgetter

import os
import datetime
import logging
import argparse 
import tqdm
import json

from utils import load_topic, load_corpus, load_results
from encoders import monoBERTCrossEncoder

def rerank(args, writer):

    reranker = monoBERTCrossEncoder(
            model_name_or_dir=args.reranker_path,
            tokenizer_name='bert-base-uncased',
            device=args.device,
            apply_softmax=False
    ).eval()

    topics = load_topics(args.topic)
    qids = list(topics.keys())
    qtexts = list(topics.values())
    corpus_texts = load_corpus(args.corpus)
    results = load_results(args.input_run, topk=args.top_k)

    for qid in qids:
        qtext = qtexts[qid]
        result = results[qid]

        # duplicated query and docs
        pairs = list([qtext] * len(result), [corpus_texts[docid] for docid in result])

        scores = reranker.predict(pairs, titles=None, max_length=args.max_length)

        hits = {result[idx]: scores[idx] for idx in range(len(scores))}            
        sorted_result = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)} 

        # write output results
        for i, (docid, score) in enumerate(sorted_result.items()):
            writier.write("{} Q0 {} {} {} CE\n".format(qid, docid, str(i+1), score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker_path", type=str, default=None)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--input_run", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    writer = open(args.output, 'w')

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)
    rerank(args, writer)
