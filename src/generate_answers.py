from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import datetime
import numpy as np

def gen_answers(model, raw_question):
    question = tokenizer(raw_question, return_tensors='pt',padding=True).to(0)
    cur_len = len(question.input_ids[0])
    # print(question)
    gen_text = model.generate(**question, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
    # print(gen_text)
    gen_text = tokenizer.batch_decode(gen_text[:,question.input_ids.shape[1]:], skip_special_tokens=True)

    print(gen_text)
    return gen_text

def get_files(q_file):
    f = open(q_file,'r')
    data = f.read()
    all_data = data.split('\n')
    return all_data

model_path = "../unlearn_models/whp_scrub/checkpoint-70"
tokenizer_path = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_path,torch_dtype=torch.float16,
).to(0).bfloat16()


tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
tokenizer.mask_token_id = tokenizer.eos_token_id
tokenizer.sep_token_id = tokenizer.eos_token_id
tokenizer.cls_token_id = tokenizer.eos_token_id

all_answers = []

print('Model and tokenizer loaded...')

list_of_files_to_load = [
    "../eval_data/whp/1_2_format/paired_forget_retain_questions.txt",
    "../eval_data/whp/pure_retain/all_retain.txt",
    "../eval_data/whp/pure_forget/harry_potter_questions.txt",
]

unlearn_method = 'scrub'
num_iters = 70
list_of_files_to_save = [
    f"../eval_answers/whp/1_2_format/{unlearn_method}_answers_{num_iters}.csv",
    f"../eval_answers/whp/pure_retain/{unlearn_method}_answers_{num_iters}.csv",
    f"../eval_answers/whp/pure_forget/{unlearn_method}_answers_{num_iters}.csv",
]

for questions, save_files in zip(list_of_files_to_load, list_of_files_to_save):
    all_answers = []
    questions = get_files(questions)
    a = datetime.datetime.now()
    bs = 128
    # res = len(questions) % bs
    for i in np.arange(len(questions)//bs+1):
        if (i+1) * bs > len(questions):
            qs = questions[i*bs:len(questions)]
        else:
            qs = questions[i*bs:(i+1)*bs]
        qs = ['Answer the following question. Question: ' + q +' Answer:' for q in qs]
        answer = gen_answers(model, qs)
        all_answers.extend(answer)
    b = datetime.datetime.now()
    print(b-a)

    df = pd.DataFrame(all_answers)
    df.to_csv(save_files)