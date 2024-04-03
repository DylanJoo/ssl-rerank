# Indexing 
# python -m pyserini.index.lucene \
#     --collection BeirFlatCollection \
#     --input /home/dju/datasets/msmarco/collection \
#     --index /home/dju/indexes/msmarco.lucene \
#     --generator DefaultLuceneDocumentGenerator \
#     --threads 4

# Searching
# python retrieval/bm25.py \
#     --k 1000 --k1 0.82 --b 0.68 \
#     --index /home/dju/indexes/msmarco.lucene \
#     --topic /home/dju/datasets/msmarco/queries.dev-subset.txt \
#     --batch_size 32 \
#     --output runs/bm25/run.msmarco-dev-subset.bm25 \

# Evaluation
echo -ne "msmarco-dev-subset | "
~/trec_eval-9.0.7/trec_eval \
    -c -m ndcg_cut.10 -m recall.100 \
    /home/dju/datasets/msmarco/qrels.msmarco-passage.dev-subset.txt \
    runs/bm25/run.msmarco-dev-subset.bm25.txt \
    | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'

echo done
