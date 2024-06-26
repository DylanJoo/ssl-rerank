import os
import torch
from transformers import Trainer
from transformers.utils import logging
from transformers.modeling_utils import unwrap_model
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

class TrainerBase(Trainer):

    def set_tokenizer(self, tokenizer=None):
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs, steps=self.state.global_step)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.state.global_step % 10 == 0:
            logger.info(f"loss: {outputs['loss'].item()} | acc: {outputs['acc']}")
            self.log({"loss": outputs['loss'].item(), "acc": outputs['acc'].item()})
            if outputs.get('losses', None):
                for k, v in outputs['losses'].items():
                    logger.info(f"{k}: {v.item()}")
                    self.log({f"{k}": v.item()})

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, **kwargs):
        """ Discard the original argument of `state_dict`, since it's from entire wrapped model.
        """
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}. The model checkpoint is an encoder for huggingface not a wrapping model.")

        model = self.model.get_encoder()
        self.model.encoder.save_pretrained(
            output_dir, state_dict=model.state_dict(), safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
