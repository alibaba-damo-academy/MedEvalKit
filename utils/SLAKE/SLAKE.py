import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm


from mathruler.grader import extract_boxed_content
from ..utils import save_json,extract,judger,get_compare_messages,judge_open_end_vqa,judge_judgement,judge_close_end_vqa
from ..base_dataset import BaseDataset

from ..question_formats import get_close_ended_prompt,get_open_ended_prompt

class SLAKE(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))


        if self.dataset_path is None:
            # dataset_path not provided by users, then download to './datas/MedXpertQA'
            self.hf_path = "BoKelvin/SLAKE"
            self.dataset_path = "./datas/SLAKE"
        else:
            # dataset_path provided by users, use the provided path directly
            self.hf_path = self.dataset_path

    
    def load_data(self):
        self.maybe_download_dataset()
        dataset_path = self.dataset_path
        dataset = []
        test_json_path = os.path.join(dataset_path,"test.json")
        with open(test_json_path,"r", encoding='utf-8') as f:
            datas = json.load(f)
        for data in datas:
            img_path = data["img_name"]
            question = data["question"]
            answer = data["answer"]
            answer_type = data["answer_type"]
            lang = data["q_lang"]

            is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
            if answer_type == "OPEN":
                prompt = get_open_ended_prompt(question,is_reasoning,lang)
            else:
                prompt = get_close_ended_prompt(question,is_reasoning,lang)

            img_path = os.path.join(dataset_path,"imgs",img_path)
            image = Image.open(img_path)
            dataset.append({"image":image,"lang":lang,"answer_type":answer_type,"answer":answer,"question":question,"prompt":prompt})
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        prompt = sample["prompt"]
        image = sample["image"]
        messages = {"prompt":prompt,"image":image}
        sample["messages"] = messages

        del sample["image"]
        return sample


    def cal_metrics(self,out_samples):
        messages_list = []

        langs = []
        answer_types = []

        metrics = {
            "total metrics" : {
                "total":0,
                "right":0
            },
            "open" : {
                "total" : 0,
                "right" : 0,
                "bleu1" : 0,
                "bleu2" : 0,
                "bleu3" : 0,
                "bleu4" : 0,
                "rouge1" : 0,
                "rouge2" : 0,
                "rougel" : 0,
                "precision" : 0,
                "recall" : 0,
                "f1" : 0,
                "em" : 0,
            },
            "close" : {
                "total" : 0,
                "right" : 0
            }
        }
        for i,out_sample in tqdm(enumerate(out_samples)):
            response = out_sample["response"]
            if extract_boxed_content(response)!= "None":
                response = extract_boxed_content(response)
            elif "<answer>" in response:
                response = extract(response,"answer")

            answer = out_sample["answer"]
            question = out_sample["question"]
            lang = out_sample["lang"]
            answer_type = out_sample["answer_type"]

            if os.environ.get("use_llm_judge","False") == "True":
                messages = get_compare_messages(question,response,answer)
                messages_list.append(messages)
                langs.append(lang)
                answer_types.append(answer_type)

            if answer_type == "OPEN":
                c_metrics = judge_open_end_vqa(answer,response)
                out_samples[i]["correct"] = c_metrics["em"]
                out_samples[i]["metrics"] = c_metrics
                metrics["total metrics"]["total"] += 1
                metrics["open"]["total"] += 1   
                if c_metrics["em"]:
                    metrics["total metrics"]["right"] += 1
                    metrics["open"]["right"] += 1      
                for metric in c_metrics:
                    metrics["open"][metric] += c_metrics[metric]           
            else:
                if answer in ["yes","no"]:
                    flag = judge_judgement(answer,response)
                else:
                    flag = judge_close_end_vqa(answer,response)
                out_samples[i]["correct"] = flag
                metrics["total metrics"]["total"] += 1
                metrics["close"]["total"] += 1
                if flag:
                    metrics["total metrics"]["right"] += 1
                    metrics["close"]["right"] += 1




        if os.environ.get("use_llm_judge","False") == "True":
            metrics["open"]["right"] = 0
            llm = judger
            results = llm.generate_outputs(messages_list)
            i = 0
            for result,lang,answer_type in zip(results,langs,answer_types):
                result = extract(result,"judge")
                result = True if result == "0" else False
                out_samples[i]["correct"] = result
                i += 1
                if result:
                    if answer_type == "OPEN":
                        metrics["open"]["right"] += 1
                    else:
                        metrics["close"]["right"] += 1
                    metrics["total metrics"]["right"] += 1


        metrics["total metrics"]["acc"] = metrics["total metrics"]["right"]/metrics["total metrics"]["total"]
        metrics["open"]["acc"] = metrics["open"]["right"]/metrics["open"]["total"]
        metrics["close"]["acc"] = metrics["close"]["right"]/metrics["close"]["total"]

        for metric in metrics["open"]:
            if metric not in ["right","total"]:
                metrics["open"][metric] = metrics["open"][metric]/metrics["open"]["total"]

        return metrics,out_samples

    def maybe_download_dataset(self):
        if not os.path.exists(self.dataset_path):
            if self.chunk_idx!=0:
                raise ValueError("Chunk inference is not support for download. Try to run eval.sh insteal of eval_chunked.sh")
            from huggingface_hub import snapshot_download
            import shutil
            print("Start downloading the SLAKE dataset...")
            snapshot_download(self.hf_path, local_dir=self.dataset_path,repo_type="dataset")
            self._unzip_img_zip_local(local_path=self.dataset_path,zip_filename="imgs.zip")
            self._unzip_img_zip_local(local_path=self.dataset_path,zip_filename="KG.zip")
            shutil.rmtree(os.path.join(self.dataset_path,".cache"))
            print("Download and extraction completed successfully")
                