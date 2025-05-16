import json
import yaml
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd

def load_model(model_name_or_path, tokenizer_path=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attention_2=True
    )

    if tokenizer_path == None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, use_fast=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=False
        )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_relearn_data(min_len=50, max_len=2000, batch_size=4):
    data = []
    for line in open(f"../relearn_data/magic.txt", "r"):
        raw_text = line
        raw_text = raw_text.strip()
        if len(raw_text) > min_len:
            data.append(str(raw_text[:max_len]))
    data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    return data 

def get_wmdp_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4, torch_ds_format=False):
        
    def get_dataset(name, torch_ds_format=False):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text'][:max_len]))
        else:
            for line in open(f"../unlearn_data/{name}.jsonl", "r"):
                raw_text = json.loads(line)
                if isinstance(raw_text, dict):
                    raw_text = raw_text.get('text','')
                raw_text = raw_text.strip()
                if len(raw_text) > min_len:
                    data.append(str(raw_text[:max_len]))
        print(len(data))
        if torch_ds_format:
            data = [data[i] for i in range(len(data))]
        else:
            data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data 
        
    def corpus_to_keywords(corpus):
        if 'bio' in corpus:
            return 'bio'
        elif 'cyber' in corpus:
            return 'cyber'
        else:
            raise NotImplementedError(f"Add your keywords for {corpus} to the keywords.json file.")

    with open("../unlearn_data/keywords.json", "r") as f:
        keywords = json.load(f)

    return (
        [keywords[corpus_to_keywords(c)] for c in forget_corpora],
        [get_dataset(c, torch_ds_format=torch_ds_format) for c in forget_corpora],
        [get_dataset(c, torch_ds_format=torch_ds_format) for c in retain_corpora]
    )

def unroll_data_list(keywords_list, forget_data_list, retain_data_list, num_data):
    unlearn_data = []
    retain_data = []
    for i in range(num_data):
        topic_idx = i % len(keywords_list)
        batch_idx = i // len(keywords_list)
        unlearn_batch = forget_data_list[topic_idx][batch_idx]
        retain_batch = retain_data_list[topic_idx][batch_idx]
        unlearn_data.append(unlearn_batch)
        retain_data.append(retain_batch)
    # print(keywords_list, len(keywords_list), len(forget_data_list))
    return unlearn_data, retain_data

def get_rwku_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4, torch_ds_format=False):
        
    def get_dataset(name, torch_ds_format=False):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text'][:max_len]))
        else:
            cnt = {}
            raw_data = load_dataset("jinzhuoran/RWKU", "train_positive_llama3",split='train')
            for x in raw_data:
                if len(x['text']) > min_len:
                    if x['subject'] in cnt and cnt[x['subject']] < 100: 
                        data.append(str(x['text'][:max_len]))
                    if x['subject'] not in cnt:
                        cnt[x['subject']] = 0
                    cnt[x['subject']] += 1
        print(len(data))
        if torch_ds_format:
            data = [data[i] for i in range(len(data))]
        else:
            data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data 
    
    return (
        get_dataset(forget_corpora, torch_ds_format=torch_ds_format),
        get_dataset(retain_corpora, torch_ds_format=torch_ds_format)
    )

def get_whp_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4, torch_ds_format=False):
        
    def get_dataset(name, torch_ds_format=False):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text'][:max_len]))
        else:
            raw_data = load_dataset("yiweifu/generic_unlearn_text", split='train')
            for i,x in enumerate(raw_data):
                if i < 3522:
                    continue
                if len(x['input_text']) > min_len:
                    data.append(str(x['input_text'][:max_len]))
        print(len(data))
        if torch_ds_format:
            data = [data[i] for i in range(len(data))]
        else:
            data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data 
    
    return (
        get_dataset(forget_corpora, torch_ds_format=torch_ds_format),
        get_dataset(retain_corpora, torch_ds_format=torch_ds_format)
    )

def get_tofu_data():
    forget_data = load_dataset("locuslab/TOFU", 'forget10')["train"]
    retain_data = load_dataset("locuslab/TOFU", 'retain90')["train"]
    return (forget_data, retain_data)


def convert_tofu_data_to_model_format(question, answer, tokenizer, max_length=500):
    question_start_token, question_end_token, answer_token = "[INST] ", " [/INST]", ""
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

def convert_data_to_model_format(data, tokenizer, max_length=500):
    encoded = tokenizer(
        data, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)


    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

class TextDatasetCustom(Dataset):
    def __init__(self, forget_corpora, retain_corpora, tokenizer, ds="rwku", max_length=512, num_data=1000):
        super(TextDatasetCustom, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ds = ds
        if ds == "wmdp":
            keywords_list, forget_data_list, retain_data_list = get_wmdp_data(forget_corpora, retain_corpora, max_len=max_length, torch_ds_format=True)
            self.forget_data, self.retain_data = unroll_data_list(keywords_list, forget_data_list, retain_data_list, num_data=num_data)
        elif ds == "rwku":
            self.forget_data, self.retain_data = get_rwku_data(forget_corpora, retain_corpora, max_len=max_length, torch_ds_format=True)
        elif ds == 'whp':
            self.forget_data, self.retain_data = get_whp_data(forget_corpora, retain_corpora, max_len=max_length, torch_ds_format=True)
        elif ds == 'tofu':
            self.forget_data, self.retain_data = get_tofu_data()

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        # print(self.data)
        ret = []
        for data_type in ["forget", "retain"]:
            data = self.retain_data if data_type == "retain" else self.forget_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)

            pad_input_ids_list = []
            label_list = []
            pad_attention_mask_list = []

            if self.ds == 'tofu':
                question = data[idx]['question']
                answer = data[idx]['answer']
                converted_data = convert_tofu_data_to_model_format(question, answer, self.tokenizer, self.max_length)
            else:
                full_text = data[idx] 
                converted_data = convert_data_to_model_format(full_text, self.tokenizer, self.max_length)
            # print(converted_data)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

            ret.append([torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze()])

        return ret

class TextDatasetCustomRelearn(Dataset):
    def __init__(self, tokenizer, max_length=512):
        super(TextDatasetCustom, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = get_relearn_data(batch_size=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data)
        full_text = self.data[idx][0]
        # print(full_text)

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        converted_data = convert_data_to_model_format(full_text, self.tokenizer, self.max_length)
        pad_input_ids_list.append(converted_data[0])
        label_list.append(converted_data[1])
        pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze()

def custom_data_collator(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def custom_data_collator_relearn(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss