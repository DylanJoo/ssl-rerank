import os
import sys
from typing import Optional, Union
from transformers import HfArgumentParser
from transformers import AutoTokenizer

from ind_cropping.options import ModelOptions, DataOptions, TrainOptions
from ind_cropping.data import load_dataset, Collator

# development
from models import Splade
from models import InBatchForSplade
from trainers import TrainerBase

os.environ["WANDB_DISABLED"] = "false"

# start here
from datasets import load_dataset

def main():

    parser = HfArgumentParser((ModelOptions, DataOptions, TrainOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()

    # [Model] tokenizer, model architecture (with bi-encoders)
    tokenizer = AutoTokenizer.from_pretrained(model_opt.model_path or model_opt.model_name)
    encoder = SpladeRep.from_pretrained(model_opt.model_name, pooling='max')
    model = InBatchForSplade(model_opt, retriever=encoder, tokenizer=tokenizer)


    # [Data] train/eval datasets, collator, preprocessor
    ## [todo] bootstrap some validatoin data
    # dataset = dataset.train_test_split(test_size=3000, seed=777)
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
    
    results = trainer.train(
        resume_from_checkpoint=train_opt.resume_from_checkpoint
    )

    return results

if __name__ == '__main__':
    main()
