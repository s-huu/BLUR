import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from rouge_score import rouge_scorer
import random
import os

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

def get_answers(file_name, terminator=False):
    if terminator:
        df = pd.read_csv(file_name,lineterminator='\n')
    else:
        df = pd.read_csv(file_name)
    all_answers = df.iloc[:,1].values
    return all_answers

def calculate_rouge(all_answers_1, all_answers_2):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    print(len(all_answers_2),len(all_answers_1))
    if len(all_answers_2) > len(all_answers_1):
        random.seed(42)
        all_answers_2 = random.sample(all_answers_2, k=len(all_answers_1))
    pbar = tqdm(zip(all_answers_1, all_answers_2))
    precision = AverageMeter()
    recall = AverageMeter()
    F1 = AverageMeter()
    for a_1,a_2 in pbar:
        scores = scorer.score(a_1,a_2)
        precision.update(scores['rougeL'].precision, 1)
        recall.update(scores['rougeL'].recall, 1)
        F1.update(scores['rougeL'].fmeasure, 1)
        pbar.set_description(f"Precision: {precision.get():.6f}, Recall: {recall.get():.6f}, F1: {F1.get():.6f}")

base_forget = get_answers('../eval_answers/whp/pure_forget/original_answers.csv')
base_retain = get_answers('../eval_answers/whp/pure_retain/original_answers.csv')

unlearn_method='npo'
num_iters=75
target_combine = get_answers(f"../eval_answers/whp/1_2_format/{unlearn_method}_answers_{num_iters}.csv",terminator=True)
target_forget = get_answers(f"../eval_answers/whp/pure_retain/{unlearn_method}_answers_{num_iters}.csv",terminator=True)
target_retain = get_answers(f"../eval_answers/whp/pure_retain/{unlearn_method}_answers_{num_iters}.csv",terminator=True)


print('Calculating Forget / Combine')
calculate_rouge(base_forget, target_combine)
print('Calculating Retain / Combine')
calculate_rouge(base_retain, target_combine)
print('Calculating Forget / Forget')
calculate_rouge(base_forget, target_forget)
print('Calculating Retain / Retain')
calculate_rouge(base_retain, target_retain)