from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
from tqdm import tqdm
import argparse

from mmlu import MMLU


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, default="/work1/shengyua/unlearn_benchmark/unlearn_models/wmdp_npo_kl/checkpoint-75")
    args = parser.parse_args()
    return args

args = parse_args()

prompt_classifier = None
tok_classifier = None
print(args.model_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
).cuda()


# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
tokenizer.pad_token = tokenizer.eos_token 

class AverageMeter:
    def __init__(self):
        self.num = 0
        self.val = 0

    def update(self, val, num):
        self.val += val * num
        self.num += num

    def get(self, percentage=False):
        val = self.val / self.num * 100 if percentage else self.val / self.num
        return val


def get_all_answers(choice_change):
    answers = ["A", "B", "C", "D"]
    choice_encoding = tokenizer(answers, return_tensors='pt', add_special_tokens=False).input_ids.squeeze(1)
    ds = MMLU(choice_change=choice_change)
    ds_eval = ds.load_dataset_for_eval('test')


    all_prompts = [q['prompt'] for q in ds_eval]
    # print(ds_eval['bio'][0])
    all_true_answers = torch.Tensor([q['correct_answer'] for q in ds_eval]).to(0)
    del ds,ds_eval
    num_prompts = len(all_prompts)
    print(num_prompts)
    batch_size=3
    all_prompts = [all_prompts[i:i+batch_size] for i in range(0, len(all_prompts), batch_size)]
    all_true_answers = torch.split(all_true_answers, batch_size)
    acc = 0
    test_acc_meter = AverageMeter()
    pbar = tqdm(zip(all_prompts,all_true_answers))
    for (p,a) in pbar:
        # print(p,a)
        input_ids = tokenizer(p, return_tensors='pt',padding=True, truncation=True).to(0)
        output = model(**input_ids)
        pred = output.logits[:,-1,choice_encoding].argmax(-1)
        test_acc = pred.eq(a).sum()/len(pred)
        test_acc_meter.update(test_acc.item(),batch_size)
        pbar.set_description(f"Acc: {test_acc_meter.get():.6f}")
        acc += pred.eq(a).sum()
        del input_ids
        del output
        del pred
        torch.cuda.empty_cache()
    print(acc)

get_all_answers(True)
# get_all_answers(False)