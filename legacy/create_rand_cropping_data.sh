INFILE=${HOME}/datasets/test_collection/corpus.jsonl
OUTDIR=${HOME}/datasets/test_collection/rand_cropping/

mkdir -p $OUTDIR
python unsupervised_learning/rand_cropping/create_train_co.py \
    --file ${INFILE} \
    --tokenizer_name bert-base-uncased \
    --save_to ${OUTDIR}/corpus_spans.jsonl
echo done
