import os
import sys
import json
from typing import Optional, Union
from transformers import HfArgumentParser
from transformers import AutoTokenizer
from dataclasses import asdict

from trainers import TrainerBase

from ind_cropping.options import ModelOptions, DataOptions, TrainOptions
from ind_cropping.data import load_dataset, Collator

os.environ["WANDB_PROJECT"]="SSL4LERR"


def main():

    parser = HfArgumentParser((ModelOptions, DataOptions, TrainOptions))
    model_opt, data_opt, train_opt = parser.parse_args_into_dataclasses()

    # change project if needed
    if train_opt.wandb_project:
        os.environ["WANDB_PROJECT"] = train_opt.wandb_project

    # [Model] tokenizer, model architecture (with bi-encoders)
    tokenizer = AutoTokenizer.from_pretrained(model_opt.model_path or model_opt.model_name)
    tokenizer.bos_token = '[CLS]'
    tokenizer.eos_token = '[SEP]'

    from models import BiCrossEncoder, monoBERT
    encoder = monoBERT.from_pretrained(model_opt.model_name)
    model = BiCrossEncoder(
            opt=model_opt, 
            encoder=encoder, 
            tokenizer=tokenizer,
            curr_mask_ratio=0.0, # totally crossencoder
    )
    
    # [Data] train/eval datasets, collator, preprocessor
    train_dataset = load_dataset(data_opt, tokenizer)
    eval_dataset = None
    collator = Collator(
            opt=data_opt, 
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
    )

    trainer = TrainerBase(
            model=model, 
            args=train_opt,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
    )
    
    # [Training]
    trainer.train(resume_from_checkpoint=train_opt.resume_from_checkpoint)
    trainer.save_model(os.path.join(train_opt.output_dir))

    final_path = train_opt.output_dir
    if  trainer.is_world_process_zero():
        with open(os.path.join(final_path, "model_opt.json"), "w") as write_file:
            json.dump(asdict(model_opt), write_file, indent=4)
        with open(os.path.join(final_path, "data_opt.json"), "w") as write_file:
            json.dump(asdict(data_opt), write_file, indent=4)
        with open(os.path.join(final_path, "train_opt.json"), "w") as write_file:
            json.dump(train_opt.to_dict(), write_file, indent=4)

if __name__ == '__main__':
    main()
