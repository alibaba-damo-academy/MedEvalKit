import torch
import os
import json
import gc
import csv
import asyncio

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm
from .utils import RubricItem,GRADER_TEMPLATE,calculate_score,_aggregate_get_clipped_mean,parse_json_to_dict
from ..utils import save_json,extract,judger,deal_tasks
from ..base_dataset import BaseDataset

def read_jsonl(jsonl_path):
    new_datas = []
    with open(jsonl_path,"r") as f:
            datas = f.readlines()
    for line in datas:
        data = json.loads(line)
        new_datas.append(data)
    return new_datas

class HealthBench(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path is not None else './datas/HealthBench'
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))

    
    def load_data(self):
        self.maybe_download_dataset()
        dataset_path = self.dataset_path
        dataset = []
        consensus_path = os.path.join(dataset_path,"consensus_2025-05-09-20-00-46.jsonl")
        hard_path = os.path.join(dataset_path,"hard_2025-05-08-21-00-10.jsonl")
        eval_path = os.path.join(dataset_path,"2025-05-07-06-14-12_oss_eval.jsonl")
        consensus_dataset = read_jsonl(consensus_path)
        hard_dataset = read_jsonl(hard_path)
        eval_dataset = read_jsonl(eval_path)
        for data in eval_dataset:
            data["dataset_type"] = "normal"
            dataset.append(data)
        for data in consensus_dataset:
            data["dataset_type"] = "consensus"
            dataset.append(data)
        for data in hard_dataset:
            data["dataset_type"] = "hard"
            dataset.append(data)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        prompt = sample["prompt"]
        messages = {"messages":prompt}
        sample["messages"] = messages
        return sample


    def cal_metrics(self,out_samples):
        async def deal_sample(sample):
            prompt = sample["prompt"]
            response = sample["response"]
            rubrics = sample["rubrics"]
            example_tags = sample["example_tags"]
            rubric_items = [RubricItem.from_dict(d) for d in rubrics]

            conversations = prompt + [dict(content=response, role="assistant")]
            conversations = "\n\n".join(
                    [f"{m['role']}: {m['content']}" for m in conversations]
                )

            grading_response_list = []
            for rubric_item in tqdm(rubric_items):
                grader_prompt = GRADER_TEMPLATE.replace(
                    "<<conversation>>", conversations
                ).replace("<<rubric_item>>", str(rubric_item))
                messages: MessageList = [dict(content=grader_prompt, role="user")]
                while True:
                    _,grading_response = await judger.generate_output_async(0,messages,temperature=0.6)
                    if not grading_response:
                        continue
                    grading_response_dict = parse_json_to_dict(grading_response)
                    if "criteria_met" in grading_response_dict:
                        label = grading_response_dict["criteria_met"]
                        if label is True or label is False:
                            break
                    print("Grading failed due to bad JSON output, retrying...")
                grading_response_list.append(grading_response_dict)
            sample["grading_response_list"] = grading_response_list
            overall_score = calculate_score(rubric_items, grading_response_list)
            # compute the overall score
            if overall_score is None:
                sample["metrics"] = None
                sample["readable_explanation_str"] = None
                sample["rubric_items_with_grades"] = None
                return sample
            metrics = {
                "overall_score": overall_score,
            }

            # compute scores for example-level tags)
            example_tag_scores = {tag: overall_score for tag in example_tags}
            if len(example_tag_scores) != len(example_tags):
                sample["metrics"] = None
                sample["readable_explanation_str"] = None
                sample["rubric_items_with_grades"] = None
                return sample
            metrics.update(example_tag_scores)

            # compute scores for rubric-level tags
            rubric_tag_items_grades = defaultdict(list)
            for rubric_item, grading_response in zip(rubric_items, grading_response_list):
                curr_item_tags = set()  # Ensure no duplicates in a rubric item.
                for tag in rubric_item.tags:
                    rubric_tag_items_grades[tag].append((rubric_item, grading_response))
                    assert tag not in curr_item_tags
                    curr_item_tags.add(tag)

            rubric_tag_scores = {}
            for tag, items_grades in rubric_tag_items_grades.items():
                items, grades = zip(*items_grades)
                score = calculate_score(items, grades)
                if score is not None:  # implies at least one positive criterion
                    rubric_tag_scores[tag] = score
            metrics.update(rubric_tag_scores)

            # construct the list of explanations and grades
            rubric_items_with_grades = []
            readable_explanation_list = []
            for rubric_item, grading_response in zip(rubric_items, grading_response_list):
                explanation = grading_response.get("explanation", "No explanation provided")
                criteria_met = grading_response["criteria_met"]
                readable_explanation = (
                    f"[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}"
                )
                readable_explanation_list.append(readable_explanation)
                rubric_items_with_grades.append(
                    {
                        **rubric_item.to_dict(),
                        "criteria_met": criteria_met,
                        "explanation": explanation,
                    }
                )

            readable_explanation_list.sort(
                key=lambda x: x.startswith("[False]"), reverse=True
            )
            readable_explanation_str = "\n\n".join(readable_explanation_list)
            readable_explanation_str = f"\n\n{readable_explanation_str}"

            sample["metrics"] = metrics
            sample["readable_explanation_str"] = readable_explanation_str
            sample["rubric_items_with_grades"] = rubric_items_with_grades
            return sample
        
        tasks = []
        for sample in out_samples:
            tasks.append(deal_sample(sample))
        
        out_samples = asyncio.run(deal_tasks(tasks))

        metrics = _aggregate_get_clipped_mean(out_samples)

        return metrics,out_samples
    
    def maybe_download_dataset(self):
        if not os.path.exists(self.dataset_path):
            if self.chunk_idx!=0:
                raise ValueError("Chunk inference is not support for download. Try to run eval.sh insteal of eval_chunked.sh")
            self._download_file_local(local_path=self.dataset_path,url='https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl')
            self._download_file_local(local_path=self.dataset_path,url='https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl')
            self._download_file_local(local_path=self.dataset_path,url='https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl')

                