exp=""
# Indexing 
## this should be done by multi-parallel process
# python encode/dense.py input \
#     --corpus /home/dju/datasets/msmarco/collection \
#     --fields text \
#     --shard-id 0 \
#     --shard-num 1 output \
#     --embeddings /home/dju/indexes/msmarco-contriever.faiss \
#     --to-faiss encoder \
#     --encoder-class contriever \
#     --encoder facebook/contriever \
#     --fields text \
#     --batch 32 \
#     --max-length 256 \
#     --device cuda

# python -m pyserini.index.merge_faiss_indexes \
#     --prefix /home/dju/indexes/msmarco-contriever.faiss \
#     --shard-num 4

# mv /home/dju/indexes/msmarco-contriever.faissfull /home/dju/indexes/msmarco-contriever.faiss

# Searching
# python retrieval/dense.py \
#     --k 1000  \
#     --index /home/dju/indexes/msmarco-contriever.faiss \
#     --encoder_path facebook/contriever \
#     --topic /home/dju/datasets/msmarco/queries.dev-subset.txt \
#     --batch_size 32 \
#     --device cuda \
#     --output runs/${exp}contriever/run.msmarco-dev-subset.contriever.txt

# Evaluation
echo -ne "msmarco-dev-subset | "
~/trec_eval-9.0.7/trec_eval \
    -c -m ndcg_cut.10 -m recall.100 \
    /home/dju/datasets/msmarco/qrels.msmarco-passage.dev-subset.txt \
    runs/${exp}contriever/run.msmarco-dev-subset.contriever.txt \
    | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'

echo done
