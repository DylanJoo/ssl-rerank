encoder=facebook/contriever
index_dir=/home/dju/indexes
data_dir=/home/dju/datasets/lotte

for dataset in science writing lifestyle recreation technology;do

    for split in test;do

        # echo indexing...${dataset}
        # python encode/dense.py input \
        #     --corpus ${data_dir}/${dataset}/${split}/collection \
        #     --fields text \
        #     --shard-id 0 \
        #     --shard-num 1 output \
        #     --embeddings ${index_dir}/lotte-${dataset}-${split}-contriever.faiss \
        #     --to-faiss encoder \
        #     --encoder-class contriever \
        #     --encoder ${encoder} \
        #     --fields text \
        #     --batch 32 \
        #     --max-length 300 \
        #     --device cuda
        #
        # echo searching...${dataset}
        # for query_type in search forum;do
        #     python retrieval/dense.py \
        #         --k 1000  \
        #         --index ${index_dir}/lotte-${dataset}-${split}-contriever.faiss \
        #         --encoder_path ${encoder} \
        #         --topic ${data_dir}/${dataset}/${split}/questions.${query_type}.tsv \
        #         --batch_size 64 \
        #         --device cuda \
        #         --output runs/baseline_contriever/run.lotte-${dataset}-${split}.${query_type}.contriever.txt
        # done

        echo evaluation...${dataset}
        for query_type in search forum;do
            echo -ne "lotte-${dataset}-${split}.${query_type} | "
            ~/trec_eval-9.0.7/trec_eval \
                -c -m ndcg_cut.10 -m recall.100 \
                ${data_dir}/${dataset}/${split}/qrels.lotte-${dataset}-${split}.${query_type}.txt \
                runs/baseline_contriever/run.lotte-${dataset}-${split}.${query_type}.contriever.txt \
                | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
        done
    done
done
