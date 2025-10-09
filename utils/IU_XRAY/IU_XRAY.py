import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

import numpy as np

from ..utils import save_json,extract
from ..base_dataset import BaseDataset

from ..question_formats import get_report_generation_prompt

class IU_XRAY(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path is not None else './datas/IU_XRAY'
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    
    def load_data(self):
        self.maybe_download_dataset()
        dataset_path = self.dataset_path
        json_path = os.path.join(dataset_path,"test.json")

        with open(json_path,"r") as f:
            dataset = json.load(f)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                if sample["findings"].strip() == "" and sample["impression"].strip() == "":
                    continue
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        print("total samples number:", len(self.samples))
        return self.samples

    def construct_messages(self,sample):
        image_root = os.path.join(self.dataset_path,"images")
        images = sample["image"]
        images = [Image.open(os.path.join(image_root,image)) for image in images]
        findings = sample["findings"]
        impression = sample["impression"]

        findings = "None" if findings.strip() == "" else findings
        impression = "None" if impression.strip() == "" else impression
        
        prompt = get_report_generation_promp()

        messages = {"prompt":prompt,"images":images}
        sample["messages"] = messages
        return sample


    def cal_metrics(self,out_samples):
        import pandas as pd

        predictions_data = []
        ground_truth_data = []

        for i,sample in enumerate(out_samples):
            response = sample["response"]
            findings = sample["findings"]
            impression = sample["impression"]
            golden = f"Findings: {findings} Impression: {impression}."

            # 生成唯一的study_id
            study_id = f"study_{i+1}"
            
            # 添加预测数据
            predictions_data.append({
                'study_id': study_id,
                'report': response
            })

            # 添加真实标签数据
            ground_truth_data.append({
                'study_id': study_id,
                'report': golden
            })


        # 创建DataFrame
        predictions_df = pd.DataFrame(predictions_data)
        ground_truth_df = pd.DataFrame(ground_truth_data)

        prediction_path = os.path.join(self.output_path,'predictions.csv')
        ground_truth_path = os.path.join(self.output_path,'ground_truth.csv')
        # 保存为CSV文件
        predictions_df.to_csv(prediction_path, index=False)
        ground_truth_df.to_csv(ground_truth_path, index=False)

        return {"total metrics":"please use cal_report_metrics.py to generate metrics"},out_samples
    
    def maybe_download_dataset(self):
        img_path = os.path.join(self.dataset_path, "images")
        if not os.path.exists(img_path):
            if self.chunk_idx!=0:
                raise ValueError("Chunk inference is not support for download. Try to run eval.sh insteal of eval_chunked.sh")

            print("Start downloading the IU_XRAY dataset...")
            url="https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
            self._download_file_local(local_path=self.dataset_path,url=url)
            self._unzip_img_zip_local(local_path=self.dataset_path,zip_filename="images.zip")
            print("Download and extraction completed successfully")

                