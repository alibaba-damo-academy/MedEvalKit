import torch
import os
import json
import gc

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,extract,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt

class OmniMedVQA(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path is not None else './datas/OmniMedVQA'
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    
    def run_model(self,samples, model, num_chunks=1, save_json_path=None):
        out_samples = []
        with torch.no_grad():
            messages_list = []
            current_messages = []
            current_samples = []
            for sample in tqdm(samples):
                messages = sample["messages"]
                current_messages.append(messages)
                current_samples.append(sample)
                if len(current_messages) >= 300:
                    messages_list.append([current_messages,current_samples])
                    current_messages = []
                    current_samples = []
            if current_messages:
                messages_list.append([current_messages,current_samples])
            
            if num_chunks > 1:
                for current_messages,current_samples in tqdm(messages_list):
                    for i in range(len(current_messages)):
                        current_messages[i]["image"] = Image.open(current_messages[i]["image"])
                    outputs = model.generate_outputs(current_messages)
                    for sample,response in zip(current_samples,outputs):
                        sample["messages"]["image"].close()
                        del sample["messages"]
                        sample["response"] = response
                        out_samples.append(sample)   
                    gc.collect()
            else:
                f = open(save_json_path,"w")
                for current_messages,current_samples in tqdm(messages_list):
                    for i in range(len(current_messages)):
                        current_messages[i]["image"] = Image.open(current_messages[i]["image"])
                    outputs = model.generate_outputs(current_messages)
                    for sample,response in zip(current_samples,outputs):
                        sample["messages"]["image"].close()
                        del sample["messages"]
                        sample["response"] = response
                        f.write(json.dumps(sample) + "\n")
                    gc.collect()
                f.close()
        return out_samples


    def load_data(self):
        self.maybe_download_dataset()
        dataset_path = self.dataset_path
        datasets = []
        open_json_path = os.path.join(dataset_path,"QA_information","Open-access")
        files = os.listdir(open_json_path)
        for file in tqdm(files,desc="Load Open-access data"):
            file_path = os.path.join(open_json_path,file)
            with open(file_path,"r") as f:
                datas = json.load(f)
            for data in datas:
                data["image_path"] = os.path.join(dataset_path,data["image_path"])
                datasets.append(data)
        for idx,sample in tqdm(enumerate(datasets)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        image = sample["image_path"]
        choices = []
        question = sample["question"]
        answer = sample["gt_answer"]
        for option in ["A","B","C","D"]:
            if f"option_{option}" in sample:
                choice = sample[f"option_{option}"]
                if answer == choice:
                    sample["gt_answer"] = option
                choice = f"{option}. {choice}"
                choices.append(choice)
                
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning)
            
        messages = {"prompt":prompt,"image":image}
        sample["messages"] = messages
        sample["choices"] = choices
        sample["answer"] = sample["gt_answer"]

        del sample["gt_answer"]
        return sample


    def cal_metrics(self,out_samples):
        total = 0
        right = 0
        total_question_type = defaultdict(int)
        right_question_type = defaultdict(int)
        total_modality_type = defaultdict(int)
        right_modality_type = defaultdict(int)
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            response = extract(response,"answer")
            
            choices = sample["choices"]
            answer = sample["answer"]
            question_type = sample["question_type"]
            modality_type = sample["modality_type"]
            total_question_type[question_type] += 1
            total_modality_type[modality_type] += 1
            total += 1

            correct = judge_multi_choice(choices,answer,response)
            out_samples[i]["correct"] = correct
            if  correct:
                right_question_type[question_type] += 1
                right_modality_type[modality_type] += 1
                right += 1
        
        metrics = {"total":total,"right":right,"acc":right/total}
        question_type_metrics = {}
        for key,value in total_question_type.items():
            right_cnt = right_question_type[key]
            question_type_metrics[key] = {"total":value,"right":right_cnt,"acc":right_cnt/value}
        modality_type_metrics = {}
        for key,value in total_modality_type.items():
            right_cnt = right_modality_type[key]
            modality_type_metrics[key] = {"total":value,"right":right_cnt,"acc":right_cnt/value}
        
        metrics = {"total metrics":metrics,"question type metrics":question_type_metrics,"modality type metrics":modality_type_metrics}
        return metrics,out_samples
    
    def maybe_download_dataset(self):
        if not os.path.exists(self.dataset_path):
            if self.chunk_idx!=0:
                raise ValueError("Chunk inference is not support for download. Try to run eval.sh insteal of eval_chunked.sh")
            print("Start downloading the OmniMedVQA dataset...")
            url="https://huggingface.co/datasets/foreverbeliever/OmniMedVQA/resolve/main/OmniMedVQA.zip"
            self._download_file_local(local_path=self.dataset_path,url=url)
            self._unzip_img_zip_local(local_path=os.path.dirname(self.dataset_path),zip_filename="OmniMedVQA.zip")
            print("Download and extraction completed successfully")


                