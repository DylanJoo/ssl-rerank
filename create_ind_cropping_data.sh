NSPLIT=4 #Must be larger than the number of processes used during training
OUTDIR=${HOME}/datasets/test_collection/
NPROCESS=2
INFILE=${HOME}/datasets/trec-covid/corpus.jsonl
mkdir -p ${OUTDIR}

split -a 2 -d -n l/${NSPLIT} ${INFILE} ${INFILE}

pids=()

for ((i=0;i<$NSPLIT;i++));do
    num=$(printf "%02d\n" $i);
    FILE=${INFILE}${num};
    echo $FILE

    python unsupervised_learning/ind_cropping/preprocess.py \
        --tokenizer bert-base-uncased \
        --datapath ${FILE} \
        --overwrite \
        --outdir ${OUTDIR} &

    pids+=($!);
    if (( $i % $NPROCESS == 0 ))
    then
        for pid in ${pids[@]}; do
            wait $pid
        done
    fi
done

for pid in ${pids[@]}; do
    wait $pid
done

rm -r ${INFILE}??
