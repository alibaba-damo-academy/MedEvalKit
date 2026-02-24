import torch
import os
import json

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from mathruler.grader import extract_boxed_content

from .data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG,DOMAIN_CAT2SUB_CAT
from .eval_utils import evaluate,parse_multi_choice_response, parse_open_response

from ..utils import extract


def run_model(samples, model,):
    out_samples = {}
    out_response = {}
    with torch.no_grad():
        messages_list = []
        current_messages = []
        current_samples = []
        for sample in tqdm(samples):
            messages = {"prompt":sample["final_input_prompt"],"image":sample["image"]}
            current_messages.append(messages)
            current_samples.append(sample)
            if len(current_messages) >= 2000:
                messages_list.append([current_messages,current_samples])
                current_messages = []
                current_samples = []
        if current_messages:
            messages_list.append([current_messages,current_samples])
        
        for current_messages,current_samples in tqdm(messages_list):
            outputs = model.generate_outputs(current_messages)
            for sample,response in zip(current_samples,outputs):
                if "<answer>" in response:
                    response = extract(response,"answer")
                if extract_boxed_content(response) != "None":
                    response = extract_boxed_content(response)
                if sample["question_type"] == "multiple-choice":
                    pred_ans = parse_multi_choice_response(response,sample["all_choices"],sample["index2ans"])
                else:
                    pred_ans = parse_open_response(response)
                out_samples[sample['id']] = pred_ans    
                out_response[sample['id']] = response  
    return out_samples,out_response


def eval_MMMU_test(model,dataset_path,output_path,subset):
    sub_dataset_list = []
    for subject in tqdm(DOMAIN_CAT2SUB_CAT[subset]):
        sub_dataset = load_dataset(dataset_path, subject, split="test")
        sub_dataset_list.append(sub_dataset)
    
    with open("utils/MMMU/answer.json", "r") as f:
        gold_answers = json.load(f)
    

    chunk_idx = int(os.environ.get("chunk_idx",0))
    num_chunks = int(os.environ.get("num_chunks",1))
    samples = []
    dataset = concatenate_datasets(sub_dataset_list)
    for idx,sample in tqdm(enumerate(dataset)):
        if idx % num_chunks == chunk_idx:
            sample = process_single_sample(sample)
            sample = construct_prompt(sample)
            samples.append(sample)

    if num_chunks == 1:
        results_path = os.path.join(output_path,"results.json")
        response_path = os.path.join(output_path,"response.json")
        out_samples,out_response = run_model(samples,model)
        save_json(results_path,out_samples)
        save_json(response_path,out_response)
        return evaluate(out_samples,gold_answers)

    elif num_chunks > 1:
        results_path = os.path.join(output_path,f"results_{chunk_idx}.json")
        response_path = os.path.join(output_path,f"response_{chunk_idx}.json")
        out_samples,out_response = run_model(samples,model)
        save_json(results_path,out_samples)
        save_json(response_path,out_response)


        total_results_path = os.listdir(output_path)
        total_results_path = [result for result in total_results_path if result.startswith("results_")]
        if len(total_results_path) == num_chunks:
            total_results = {}
            for result in total_results_path:
                results_path = os.path.join(output_path,result)
                with open(results_path,"r") as f:
                    partial_results = json.load(f)
                    total_results.update(partial_results)
                    os.remove(results_path)
            with open(os.path.join(output_path,"results.json"),"w") as f:
                json.dump(total_results,f)
                
            return evaluate(total_results,gold_answers)
        else:
            return
    else:
        raise ValueError("num_chunks must be greater than 0")

