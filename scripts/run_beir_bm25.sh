# for dataset in scifact trec-covid scidocs;do
#     # Indexing
#     index=/home/dju/indexes/beir/${dataset}-multifield.lucene
#     python -m pyserini.index.lucene \
#         --collection BeirMultifieldCollection \
#         --input /home/dju/datasets/beir/${dataset}/collection \
#         --index ${index} \
#         --generator DefaultLuceneDocumentGenerator \
#         --threads 4 --fields title
#
#     # Search
#     python retrieval/bm25.py \
#         --k 1000 --k1 0.9 --b 0.4 \
#         --index /home/dju/indexes/beir/${dataset}-multifield.lucene \
#         --topic /home/dju/datasets/beir/${dataset}/queries.jsonl \
#         --batch_size 8 \
#         --fields contents=1.0 title=1.0 \
#         --output runs/bm25/run.beir.${dataset}.bm25-multifield.txt
# done

# Evaluation
for dataset in scifact trec-covid scidocs;do
    echo -ne "beir-${dataset}  | " 

    ~/trec_eval-9.0.7/trec_eval \
        -c -m ndcg_cut.10 -m recall.100 \
        /home/dju/datasets/beir/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
        runs/bm25/run.beir.${dataset}.bm25-multifield.txt \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done
