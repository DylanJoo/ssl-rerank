index_dir=/home/dju/indexes/beir
data_dir=/home/dju/datasets/beir

mkdir -p runs 

for ckpt in 2500 5000;do

    for dataset in trec-covid;do
        exp=ind-cropping-bce-random
        reranker=/home/dju/ssl-rerank/models/ckpt/monobert-${exp}-${dataset}/checkpoint-${ckpt}

        echo reranking...${dataset}...${exp}
        python ranking/rerank.py \
            --reranker_path ${reranker} \
            --topic ${data_dir}/${dataset}/queries.jsonl \
            --corpus ${data_dir}/${dataset}/collection/corpus.jsonl \
            --batch_size 128 \
            --max_length 384 \
            --input_run runs/bm25/run.beir.${dataset}.bm25-multifield.txt \
            --output runs/monobert-${exp}/run.beir.${dataset}.monobert-${exp}.txt \
            --top_k 10  \
            --device cpu
    done

    for dataset in trec-covid;do
        echo -ne "beir-${dataset}.${exp}.${ckpt}  | " 
        ~/trec_eval-9.0.7/trec_eval \
            -c -m ndcg_cut.10 -m recall.100 \
            ${data_dir}/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
            runs/monobert-${exp}/run.beir.${dataset}.monobert-${exp}.txt \
            | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
    done

done
