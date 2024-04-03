dataset_dir=/home/dju/datasets

# Analyze msmarco and other datasets
# Unigram (the same setting as DAPT)
python tools/analysis/vocab_overlap.py \
    --files_path ${dataset_dir}/msmarco/collection/corpus.jsonl \
    --files_path ${dataset_dir}/beir/scifact/collection/corpus.jsonl \
    --files_path ${dataset_dir}/beir/scidocs/collection/corpus.jsonl \
    --files_path ${dataset_dir}/beir/trec-covid/collection/corpus.jsonl \
    --output_image analysis/1gram-beir.png \
    --output_text analysis/1gram-beir.txt \
    --min_df 50 --max_ngram 1

python tools/analysis/vocab_overlap.py \
    --files_path ${dataset_dir}/msmarco/collection/corpus.jsonl \
    --files_path ${dataset_dir}/lotte/lifestyle/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/recreation/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/science/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/technology/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/writing/test/collection/docs00.json \
    --output_image analysis/1gram-lotte.png  \
    --output_text analysis/1gram-lotte.txt \
    --min_df 50 --max_ngram 1

# Unigram and Bi-gram 
python tools/analysis/vocab_overlap.py \
    --files_path ${dataset_dir}/msmarco/collection/corpus.jsonl \
    --files_path ${dataset_dir}/beir/scifact/collection/corpus.jsonl \
    --files_path ${dataset_dir}/beir/scidocs/collection/corpus.jsonl \
    --files_path ${dataset_dir}/beir/trec-covid/collection/corpus.jsonl \
    --output_image analysis/2gram-beir.png \
    --output_text analysis/2gram-beir.txt \
    --min_df 50 --max_ngram 2

python tools/analysis/vocab_overlap.py \
    --files_path ${dataset_dir}/msmarco/collection/corpus.jsonl \
    --files_path ${dataset_dir}/lotte/lifestyle/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/recreation/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/science/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/technology/test/collection/docs00.json \
    --files_path ${dataset_dir}/lotte/writing/test/collection/docs00.json \
    --output_image analysis/2gram-lotte.png  \
    --output_text analysis/2gram-lotte.txt \
    --min_df 50 --max_ngram 2

