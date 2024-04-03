import os
import sys
from typing import Optional, Union
from transformers import HfArgumentParser
from transformers import AutoTokenizer

# replace the argument parser to options
# from arguments import ModelArgs, DataArgs, TrainArgs 
from ind_cropping.options import ModelOptions, DataOptions, TrainOptions
from ind_cropping.data import load_dataset, Collator

from models import Contriever
from models._dev import Contriever
from models import InBatch
from trainers import TrainerBase


os.environ["WANDB_DISABLED"] = "false"

def main():

    parser = HfArgumentParser((ModelOptions, DataOptions, TrainOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()

    # [Model] tokenizer, model architecture (with bi-encoders)
    tokenizer = AutoTokenizer.from_pretrained(model_opt.model_path or model_opt.model_name)
    encoder = Contriever.from_pretrained(model_opt.model_name, pooling=model_opt.pooling)
    model = InBatch(model_opt, retriever=encoder, tokenizer=tokenizer)
    
    # [Data] train/eval datasets, collator, preprocessor
    train_dataset = load_dataset(data_opt, tokenizer)
    eval_dataset = None
    collator = Collator(opt=data_opt)

    trainer = TrainerBase(
            model=model, 
            args=train_opt,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
    )
    
    # ***** strat training *****
    results = trainer.train(resume_from_checkpoint=train_opt.resume_from_checkpoint)

    return results

if __name__ == '__main__':
    main()
