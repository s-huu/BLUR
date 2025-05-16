import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import pandas as pd
from tqdm import tqdm

def get_questions(q_file):
    f = open(q_file,'r')
    data = f.read()
    all_data = data.split('\n')
    return all_data

def get_answers(file_name, terminator=False):
    if terminator:
        df = pd.read_csv(file_name,lineterminator='\n')
    else:
        df = pd.read_csv(file_name)
    all_answers = df.iloc[:,1].values
    return all_answers

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

def generate_and_calculate_perplexity(model, tokenizer, input_text, output_text):


    full_text = input_text + output_text
    encoded_full = tokenizer(
        full_text, 
        return_tensors='pt',
    ).to(0)
    encoded_inp = tokenizer(
        input_text, 
        return_tensors='pt',
    ).to(0)
    inp_len = encoded_inp.input_ids.shape[1]
    
    # Forward pass with the generated text to get logits
    with torch.no_grad():
        outputs = model(**encoded_full)
        logits = outputs.logits
    
    # Shift to get next token predictions
    shift_logits = logits[:, inp_len:-1, :].contiguous()
    shift_labels = encoded_full.input_ids[:, (1+inp_len):].contiguous()
    
    # Calculate loss with CrossEntropyLoss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Perplexity = exp(loss)
    perplexity = math.exp(loss.item())
    
    return  perplexity

# tokenizer_path = "meta-llama/Llama-2-7b-hf"
# model_path = "meta-llama/Llama-2-7b-hf"
# tokenizer_path = "meta-llama/Meta-Llama-3-8b-Instruct"
# model_path = "meta-llama/Meta-Llama-3-8b-Instruct"
tokenizer_path = "HuggingFaceH4/zephyr-7b-beta"
model_path = "HuggingFaceH4/zephyr-7b-beta"

# model_path = "/work1/shengyua/unlearn_benchmark/unlearn_models/whp_npo_kl/checkpoint-134"

model = AutoModelForCausalLM.from_pretrained(
    model_path,torch_dtype=torch.float16,
).to(0).bfloat16()


tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path
)
tokenizer.pad_token_id = tokenizer.eos_token_id

list_of_questions_to_load = [
    "/work1/shengyua/unlearn_benchmark/eval_data/wmdp/1_2_format/paired_forget_retain_questions.txt",
    "/work1/shengyua/unlearn_benchmark/eval_data/wmdp/pure_retain/combined_retain_questions.txt",
    "/work1/shengyua/unlearn_benchmark/eval_data/wmdp/pure_forget/all_forget.txt",
]

unlearn_method = 'scrub'
num_iters = 187
list_of_answers_to_load = [
    f"/work1/shengyua/unlearn_benchmark/eval_answers/wmdp/1_2_format/{unlearn_method}_answers_{num_iters}.csv",
    f"/work1/shengyua/unlearn_benchmark/eval_answers/wmdp/pure_retain/{unlearn_method}_answers_{num_iters}.csv",
    f"/work1/shengyua/unlearn_benchmark/eval_answers/wmdp/pure_forget/{unlearn_method}_answers_{num_iters}.csv",
]

# list_of_questions_to_load = [
#     "/work1/shengyua/unlearn_benchmark/eval_data/rwku/1_2_format/paired_forget_retain_questions.txt",
#     "/work1/shengyua/unlearn_benchmark/eval_data/rwku/pure_retain/all_retain.txt",
#     "/work1/shengyua/unlearn_benchmark/eval_data/rwku/pure_forget/rwku_questions.txt",
# ]

# unlearn_method = 'scrub'
# num_iters = 425
# # list_of_answers_to_load = [
# #     f"/work1/shengyua/unlearn_benchmark/eval_answers/rwku/1_2_format/{unlearn_method}_answers_{num_iters}.csv",
# #     f"/work1/shengyua/unlearn_benchmark/eval_answers/rwku/pure_retain/{unlearn_method}_answers_{num_iters}.csv",
# #     f"/work1/shengyua/unlearn_benchmark/eval_answers/rwku/pure_forget/{unlearn_method}_answers_{num_iters}.csv",
# # ]
# list_of_answers_to_load = [
#     f"/work1/nkale/unlearn_benchmark/eval_answers/rwku/{unlearn_method}/checkpoint-{num_iters}/1_2_answers.csv",
#     f"/work1/nkale/unlearn_benchmark/eval_answers/rwku/{unlearn_method}/checkpoint-{num_iters}/new_retain_answers.csv",
#     f"/work1/nkale/unlearn_benchmark/eval_answers/rwku/{unlearn_method}/checkpoint-{num_iters}/forget_answers.csv",
# ]

for qu,an in zip(list_of_questions_to_load, list_of_answers_to_load):
    question = get_questions(qu)
    answer = get_answers(an, True)
    # print(question, answer)
    pbar = tqdm(zip(question, answer))
    perp = AverageMeter()
    for q,a in pbar:
        q = 'Answer the following question. Question: ' + q +' Answer:'
        score = generate_and_calculate_perplexity(model, tokenizer, q, a)
        perp.update(score, 1)
        pbar.set_description(f"Perplexity: {perp.get():.6f}")


