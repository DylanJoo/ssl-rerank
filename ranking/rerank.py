from operator import itemgetter

import os
import datetime
import logging
import argparse 
from tqdm import tqdm
import json

from utils import load_topic, load_corpus, load_results, batch_iterator
from encoders import monoBERTCrossEncoder

def rerank(args, writer):

    reranker = monoBERTCrossEncoder(
            model_name_or_dir=args.reranker_path,
            tokenizer_name='bert-base-uncased',
            device=args.device,
            apply_softmax=False
    )

    topics = load_topic(args.topic)
    qids = list(topics.keys())
    # qtexts = list(topics.values())
    corpus_texts = load_corpus(args.corpus)
    results = load_results(args.input_run, topk=args.top_k)

    for qid in tqdm(qids, total=len(qids)):
        qtext = topics[qid]
        result = results[qid]
        dtexts = [corpus_texts[docid] for docid in result]

        # predict
        scores = []
        for batch_dtexts in batch_iterator(dtexts, args.batch_size):
            batch_scores = reranker.predict(
                    [[qtext] * len(batch_dtexts), batch_dtexts], 
                    titles=None, 
                    max_length=args.max_length
            )
            scores.extend(batch_scores)

        # re-order
        hits = {result[idx]: scores[idx] for idx in range(len(scores))}            
        sorted_result = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)} 

        # write
        for i, (docid, score) in enumerate(sorted_result.items()):
            writer.write("{} Q0 {} {} {} CE\n".format(qid, docid, str(i+1), score))

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

    os.makedirs(args.output.rsplit('/', 1)[0], exist_ok=True)

    writer = open(args.output, 'w')
    rerank(args, writer)
