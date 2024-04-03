# for dataset in scifact trec-covid scidocs;do
for dataset in trec-covid;do

    # Index: document pre-encoding
    mkdir -p ${HOME}/tmp-datasets/beir/${dataset}
    model_name=naver/splade-cocondenser-ensembledistil
    encoded=${HOME}/tmp-datasets/beir/${dataset}/corpus-splade_pp.jsonl
    python retrieval/sparse_pre_encode.py \
	--collection ${HOME}/datasets/beir/${dataset}/collection/corpus.jsonl \
	--encoded_output ${encoded} \
	--model_name_or_dir ${model_name} \
	--batch_size 4 \
	--max_length 256 \
	--device cpu \
	--quantization_factor 100 --debug

    # # Index: impact vector (weight bm25) 
    # index=${HOME}/indexes/beir/${dataset}-splade_pp.impact
    # python -m pyserini.index.lucene \
	# --collection JsonVectorCollection \
	# --input ${encoded} \
	# --index ${index}  \
	# --generator DefaultLuceneDocumentGenerator \
	# --threads 36 \
	# --impact --pretokenized
    #
    # # Search
    # python3 retrieval/sparse_search.py \
	# --topic ${HOME}/datasets/beir/${dataset}/queries.jsonl  \
	# --output runs/run.beir.${dataset}.splade_pp.impact.txt \
	# --index ${index} \
	# --batch_size 8 \
	# --device cuda \
	# --encoder ${model_name} \
	# --k 1000 --min_idf 0 \
done

# Evaluation
# for dataset in scifact trec-covid scidocs;do
#     echo -ne "beir-${dataset}  | " 
#
#     ~/trec_eval-9.0.7/trec_eval \
#         -c -m ndcg_cut.10 -m recall.100 \
#         /home/dju/datasets/beir/${dataset}/qrels.beir-v1.0.0-${dataset}.test.txt \
#         runs/baseline_bm25/run.beir.${dataset}.bm25-multifield.txt \
#         | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
# done
