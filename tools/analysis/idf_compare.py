# This code was cloned from: http://github.com/allenai/dont-stop-pretraining
# Reference: https://arxiv.org/pdf/2004.10964.pdf 
import sys
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse
from collections import defaultdict

sns.set(context="paper", style="white", font_scale=2.1)

def load_data(data_path, field='text'):
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

def build_vectors(file, field, min_df):
    dd = defaultdict(float)
    text = load_data(file, field)
    tfidf_vectorizer = TfidfVectorizer(
            min_df=min_df,
            stop_words="english", 
            ngram_range=(args.min_ngram, args.max_ngram)
    )
    tfidf_vectorizer.fit(text)

    # [NOTE] previous option: use the average document embeddings, 
    # it has limited insights.
    term_list = tfidf_vectorizer.get_feature_names_out()
    idf_list = tfidf_vectorizer.idf_
    dd.update({k: v for k, v in zip(term_list, idf_list)})
    return dd

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

    if '.' in args.min_df:
        number = float(args.min_df)
    else:
        number = int(float(args.min_df))

    # Pre-loaded corpora
    all_documents = []
    for path in files:
        if 'lotte' in path:
            all_documents += load_data(path, field='contents')
        else:
            all_documents += load_data(path)

    # Construct a tf-idf vectorizer with all doucments
    count_vectorizer = CountVectorizer(
            min_df=number,
            stop_words="english", 
            ngram_range=(args.min_ngram, args.max_ngram),
            binary=True # only extract the appearance
    )
    count_vectorizer.fit(tqdm(all_documents))
    all_vocabulary = list(count_vectorizer.vocabulary_.keys())
    del all_documents

    # Build tfidf vectors for the datasets
    idf = {}
    for path in files:
        kwargs = dict(file=path, field='text', min_df=number)
        if 'scidocs' in path:
            idf['SD'] = build_vectors(**kwargs)
        elif 'scifact' in path:
            idf['SF'] = build_vectors(**kwargs)
        elif 'trec-covid' in path:
            idf['TC'] = build_vectors(**kwargs)
        elif 'msmarco' in path:
            idf['MS'] = build_vectors(**kwargs)
        elif 'lotte' in path:
            kwargs['field'] = 'contents'
            if 'lifestyle' in path:
                idf['lotte-li'] = build_vectors(**kwargs)
            elif 'recreation' in path:
                idf['lotte-re'] = build_vectors(**kwargs)
            elif 'science' in path:
                idf['lotte-sc'] = build_vectors(**kwargs)
            elif 'writing' in path:
                idf['lotte-wr'] = build_vectors(**kwargs)
            elif 'technology' in path:
                idf['lotte-te'] = build_vectors(**kwargs)

    if len(idf.keys()) <= 1:
        raise ValueError('At least two collections required.')

    # reorganize the array with shared vocabulary 
    idf_vector = defaultdict(list)
    for filename, idf_dict in idf.items():
        for vocab in all_vocabulary:
            idf_vector[filename] += [idf_dict[vocab]]

    # calculate the dot product of idf vector 
    idf_matrix = np.array([idf_vector[k] for k in idf_vector.keys()]) 
    idf_product = idf_matrix  @ idf_matrix.T

    with open(args.output_text, "w") as f:
        for row in idf_product:
            f.write(str(row.tolist())+'\n')

    # calculate the dot product of idf diff
    file_pairs = itertools.combinations(list(idf.keys()), 2)

    output_dict = {}
    with open(args.output_text.replace('txt', 'json'), "w") as f:
        for x, y in file_pairs:
            idf_vector_diff = np.array(idf_vector[x]) - np.array(idf_vector[y])
            idf_diff = list(zip(all_vocabulary, idf_vector_diff))
            idf_diff = {t: v for t, v in sorted(idf_diff, key=lambda x: x[1])}
            output_dict[f'{x} - {y}'] = idf_diff
        json.dump(output_dict, f)
    
    # # printing
    # labels = list(vectors.keys())
    # ax = sns.heatmap(idf_product, 
    #         cmap="Blues", xticklabels=labels, annot=True, 
    #         fmt=".1f", cbar=False, yticklabels=labels
    # )
    # #
    # plt.yticks(rotation=0)
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(args.output_image, dpi=300)
    # print('done')
