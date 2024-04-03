root_dir=/home/dju/datasets/lotte

for dataset in science writing lifestyle recreation technology;do
    for split in dev test;do
        base_dir=${root_dir}/${dataset}/${split}

        # Convert tsv to jsonl (code is cloned from Anserini)
        # tsv=${base_dir}/collection.tsv
        # data_dir=${base_dir}/collection
        #
        # if [ -e ${data_dir}/docs00.json ]
        # then
        #     echo "jsonl collection exists."
        # else
        #     python tools/convert_collection_to_jsonl.py \
        #         --collection-path ${tsv} \
        #         --output-folder ${data_dir}
        #     rm ${tsv}
        # fi

        # Indexing
        index=/home/dju/indexes/lotte-${dataset}-${split}.lucene
        # python -m pyserini.index.lucene \
        #     --collection JsonCollection \
        #     --input ${data_dir} \
        #     --index ${index} \
        #     --generator DefaultLuceneDocumentGenerator \
        #     --threads 4

        # Search 
        # for query_type in search forum;do
        #     python retrieval/bm25.py \
        #         --k 1000 --k1 0.9 --b 0.4 \
        #         --index ${index} \
        #         --topic ${base_dir}/questions.${query_type}.tsv \
        #         --batch_size 8 \
        #         --output runs/run.lotte-${dataset}-${split}.${query_type}.bm25.txt
        # done

        # Evaluation (trec) 
        for query_type in search forum;do
            ## Convert jsonl to qrels if needed
            if [ -e ${base_dir}/qrels.lotte-${dataset}-${split}.${query_type}.txt ]
            then
                echo -ne "lotte-${dataset}-${split}.${query_type} | "
            else
                python tools/convert_qas_to_qrels.py \
                    --qas-jsonl-path ${base_dir}/qas.${query_type}.jsonl \
                    --qrels-txt-path ${base_dir}/qrels.lotte-${dataset}-${split}.${query_type}.txt \
                echo -ne "lotte-${dataset}-${split}.${query_type} | "
            fi

            ~/trec_eval-9.0.7/trec_eval \
                -c -m ndcg_cut.10 -m recall.100 \
                ${base_dir}/qrels.lotte-${dataset}-${split}.${query_type}.txt \
                runs/run.lotte-${dataset}-${split}.${query_type}.bm25.txt \
                | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
        done
    done
done

# Evaluation (Lotte) --> S@K
# for split in dev test;do
#     echo "[Lotte-${split}]"
#     python tools/evaluate_lotte_rankings.py \
#         --k 5 --split ${split} \
#         --data_dir ${root_dir} \
#         --rankings_dir runs
# done
