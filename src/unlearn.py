import torch
from utils import load_model, TextDatasetCustom, custom_data_collator
from trainers import CustomTrainerGAAndNPO, CustomTrainerScrub
import hydra 
import transformers
import os
from pathlib import Path
from omegaconf import OmegaConf
    
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="unlearn_wmdp_scrub")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    model_path = cfg.model_path
    print(model_path)
    # model, tokenizer = load_model(model_path, tokenizer_path='NousResearch/Llama-2-7b-chat-hf')
    # oracle_model, _ = load_model(model_path, tokenizer_path='NousResearch/Llama-2-7b-chat-hf')
    model, tokenizer = load_model(model_path)
    oracle_model, _ = load_model(model_path)

    forget_corpora = "bio-forget-corpus,cyber-forget-corpus-safe"
    retain_corpora = "wikitext,wikitext"
    retain_corpora = retain_corpora.split(",")
    forget_corpora = forget_corpora.split(",")

    # forget_corpora = "rwku"
    # retain_corpora = "wikitext"

    max_length = 500
    torch_format_dataset = TextDatasetCustom(forget_corpora=forget_corpora, retain_corpora=retain_corpora,tokenizer=tokenizer, ds="wmdp", max_length=max_length)


    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    # --nproc_per_node gives the number of GPUs per = num_devices. take it from torchrun/os.environ
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    print(f"Len dataset: {len(torch_format_dataset)}")
    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(2, max_steps//10),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            evaluation_strategy="no",
            deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay
        )
    print("Loading from checkpoint")
    model.gradient_checkpointing_enable()
    
    if cfg.forget_loss in ['grad_ascent','npo']:
        trainer = CustomTrainerGAAndNPO(
            model=model,
            oracle_model=oracle_model,
            forget_loss=cfg.forget_loss,
            with_kl=cfg.with_kl,
            train_dataset=torch_format_dataset,
            eval_dataset=torch_format_dataset,
            args=training_args,
            data_collator=custom_data_collator,
        )
    elif cfg.forget_loss == 'scrub':
        trainer = CustomTrainerScrub(
            model=model,
            eps1=0.001,
            eps2=0.1,
            scrub_max_epochs=4,
            forget_loss=cfg.forget_loss,
            oracle_model=oracle_model,
            train_dataset=torch_format_dataset,
            eval_dataset=torch_format_dataset,
            args=training_args,
            data_collator=custom_data_collator,
        )
    else:
        raise Exception("Loss type not supported")

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    oracle_model.config.use_cache = False
    trainer.train()

    # model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()

