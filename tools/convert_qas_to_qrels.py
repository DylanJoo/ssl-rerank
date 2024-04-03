import jsonlines
import os
import argparse

def convert_qas_jsonl(args):
    print('Converting qas_jsonl...')

    output_qrels_file = open(args.qrels_txt_path, 'w', encoding='utf-8', newline='\n')
    with jsonlines.open(args.qas_jsonl_path, mode="r") as f:

        for line in f:
            qid = int(line["qid"])
            answer_pids = set(line["answer_pids"])

            for pid in answer_pids:
                output_qrels_file.write(f'{qid} 0 {pid} 1\n')

    output_qrels_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Lotte qas jsonl into trec qrels files for evaluation')
    parser.add_argument('--qas-jsonl-path', required=True)
    parser.add_argument('--qrels-txt-path', required=True)
    args = parser.parse_args()

    convert_qas_jsonl(args)
    print('Done!')
