import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt

class PMC_VQA(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path 
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))

        if self.dataset_path is None:
            # dataset_path not provided by users, then download to './datas/MedXpertQA'
            self.hf_path = "RadGenome/PMC-VQA"
            self.dataset_path = "./datas/PMC-VQA"
        else:
            # dataset_path provided by users, use the provided path directly
            self.hf_path = self.dataset_path

    
    def load_data(self):
        self.maybe_download_dataset()
        dataset_path = self.dataset_path
        dataset = []
        csv_path = os.path.join(dataset_path,"test_2.csv")
        reader = csv.reader(open(csv_path, 'r', encoding='utf-8'))
        next(reader)
        for i,row in enumerate(reader):

            index,figure_path,caption,question,choiceA,choiceB,choiceC,choiceD,answer,split = row
            choices = [choiceA,choiceB,choiceC,choiceD]
            is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
            prompt = get_multiple_choice_prompt(question,choices,is_reasoning)
            image_path = os.path.join(dataset_path,"figures",figure_path)
            sample = {"prompt":prompt,"answer":answer,"image":image_path,"choices":choices}
            dataset.append(sample)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        prompt = sample["prompt"]
        image = sample["image"] if os.path.exists(sample["image"]) else os.path.join(self.dataset_path, "images", sample["image"])
        messages = {"prompt":prompt,"image":image}
        sample["messages"] = messages
        del sample["image"]
        return sample


    def cal_metrics(self,out_samples):
        total = 0
        right = 0
        total_modality = defaultdict(int)
        right_modality = defaultdict(int)
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            choices = sample["choices"]
            answer = sample["answer"]

            correct = judge_multi_choice(choices,answer,response)
            out_samples[i]["correct"] = correct
            if correct:
                right += 1
            total += 1

        metrics = {"total metrics":{"total":total,"right":right,"acc":right/total}}
        return metrics,out_samples
    
    def maybe_download_dataset(self):
        if not os.path.exists(self.dataset_path):
            if self.chunk_idx!=0:
                raise ValueError("Chunk inference is not support for download. Try to run eval.sh insteal of eval_chunked.sh")
            from huggingface_hub import snapshot_download
            import shutil
            print("Start downloading the PMC-VQA dataset...")
            snapshot_download(self.hf_path, local_dir=self.dataset_path,repo_type="dataset")
            images_zip_path = os.path.join(self.dataset_path, "images.zip")
            images2_zip_path = os.path.join(self.dataset_path, "images_2.zip")
            self._unzip_img_zip_local(local_path=self.dataset_path,zip_filename="images.zip")
            os.rename(images2_zip_path, images_zip_path)
            self._unzip_img_zip_local(local_path=self.dataset_path,zip_filename="images.zip")
            shutil.rmtree(os.path.join(self.dataset_path,".cache"))
            print("Download and extraction completed successfully")




        



                