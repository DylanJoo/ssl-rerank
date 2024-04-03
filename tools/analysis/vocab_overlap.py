# This code was cloned from: http://github.com/allenai/dont-stop-pretraining
# Reference: https://arxiv.org/pdf/2004.10964.pdf 
import sys
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import json
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse

sns.set(context="paper", style="white", font_scale=2.1)

def load_data(data_path, field):
    examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            line = line.strip()
            if line:
                if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                    example = json.loads(line)
                else:
                    example = {field: line}
                text = example[field]
                examples.append(text)
    return examples

def load_vocab(file, field, min_df, ngram_range):
    text = load_data(file, field)
    count_vectorizer = CountVectorizer(
            min_df=min_df, 
            stop_words="english", 
            ngram_range=ngram_range
    )
    count_vectorizer.fit(tqdm(text))
    vocab = set(count_vectorizer.vocabulary_.keys())
    return vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", nargs="+", action='append')
    parser.add_argument("--output_image", default='dev.png')
    parser.add_argument("--output_text", default='dev.txt')
    parser.add_argument("--min_df", default=3, type=str)
    parser.add_argument("--min_ngram", default=1, type=int)
    parser.add_argument("--max_ngram", default=1, type=int)
    args = parser.parse_args()
    files = [p[0] for p in args.files_path]

    vocabs = {}
    if '.' in args.min_df:
        number = float(args.min_df)
    else:
        number = int(float(args.min_df))

    vectorizer_args = {
            'min_df': number,
            'ngram_range': (args.min_ngram, args.max_ngram)
    }

    for path in files:
        if 'scidocs' in path:
            vocabs['SD'] = load_vocab(path, 'text', **vectorizer_args)
        elif 'scifact' in path:
            vocabs['SF'] = load_vocab(path, 'text', **vectorizer_args)
        elif 'trec-covid' in path:
            vocabs['TC'] = load_vocab(path, 'text', **vectorizer_args)
        elif 'msmarco' in path:
            vocabs['MS'] = load_vocab(path, 'text', **vectorizer_args)
        elif 'lotte' in path:
            if 'lifestyle' in path:
                vocabs['lotte-li'] = load_vocab(path, 'contents', **vectorizer_args)
            elif 'recreation' in path:
                vocabs['lotte-re'] = load_vocab(path, 'contents', **vectorizer_args)
            elif 'science' in path:
                vocabs['lotte-sc'] = load_vocab(path, 'contents', **vectorizer_args)
            elif 'writing' in path:
                vocabs['lotte-wr'] = load_vocab(path, 'contents', **vectorizer_args)
            elif 'technology' in path:
                vocabs['lotte-te'] = load_vocab(path, 'contents', **vectorizer_args)

    if len(vocabs.keys()) <= 1:
        raise ValueError('At least two collections required.')

    file_pairs = itertools.combinations(list(vocabs.keys()), 2)
    overlaps = {}
    for x, y in file_pairs:
        intersection = vocabs[x] & vocabs[y]
        union = (vocabs[x] | vocabs[y])
        overlaps[x + "_" + y] = len(intersection) / len(union)
        with open(args.output_text, "w+") as f:
            json.dump(overlaps, f)
    
    data = []

    # printing
    z = {}
    for key in overlaps.keys():
        file_1, file_2 = key.split('_')
        if not z.get(file_1):
            z[file_1] = {}
        z[file_1][file_2] = overlaps[key]
        if not z.get(file_2):
            z[file_2] = {}
        z[file_2][file_1] = overlaps[key]

    labels = list(vocabs.keys())
     
    for ix, key in enumerate(labels):
        items = []
        for subkey in labels:
            if not z[key].get(subkey):
                items.append(1.0)
            else:
                items.append(z[key][subkey])
        data.append(items)

    data = np.array(data) * 100
    ax = sns.heatmap(data, cmap="Blues", vmin=30, xticklabels=labels, annot=True, fmt=".1f", cbar=False, yticklabels=labels)

    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(args.output_image, dpi=300)
