from openai import AzureOpenAI, OpenAI,AsyncAzureOpenAI,AsyncOpenAI

from abc import abstractmethod
from tqdm.asyncio import tqdm_asyncio 
from tqdm import tqdm
import os
import logging
import asyncio
import base64

import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)


from PIL import Image
from io import BytesIO


def reduce_image_size(image, max_size_mb=5):
    """递归压缩图片直到小于指定大小"""
    quality = 100
    width, height = image.size
    
    while True:
        # 保存图片到内存中以检查大小
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        size_mb = len(buffered.getvalue()) / (1024 * 1024)  # 转换为MB
        
        # 如果大小已经小于目标大小，返回当前图片
        if size_mb <= max_size_mb:
            return image
            
        # 如果质量已经很低但仍然太大，则缩小尺寸
        width = int(width * 0.9)
        height = int(height * 0.9)
        image = image.resize((width, height), Image.LANCZOS)


def encode_image(image_path, max_size_mb=5):
    """编码图片为base64，并确保大小不超过指定值"""
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
        
    image = image.convert("RGB")
    
    # 检查原始图片大小
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    size_mb = len(buffered.getvalue()) / (1024 * 1024)
    
    # 如果超过最大大小，进行压缩
    if size_mb > max_size_mb:
        image = reduce_image_size(image, max_size_mb)
    
    # 保存处理后的图片
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def before_retry_fn(retry_state):
    if retry_state.attempt_number % 2 == 0:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")

def log_retry_error(retry_state):
    """保留原始异常信息"""
    exception = retry_state.outcome.exception()
    print(f"Original error: {exception}")
    return exception

class openai_llm:
    def __init__(self,model,args) -> None:
        self.model = model

        api_key = os.environ["api_key"]
        bsae_url = os.environ["base_url"]

        self.client = OpenAI(
            api_key=api_key,
            base_url= bsae_url
            )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url= bsae_url
            )
        
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_tokens = args.max_new_tokens
    
    
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(100), before=before_retry_fn,retry_error_callback=log_retry_error)
    def response(self,messages,**kwargs):
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature= self.temperature,
            max_tokens=self.max_tokens,
            timeout=kwargs.get("timeout", 60)
        )
        return response.choices[0].message.content
    
    
    @retry(wait=wait_fixed(20), stop=stop_after_attempt(50), before=before_retry_fn,retry_error_callback=log_retry_error)
    async def response_async(self,messages,**kwargs):
        response = await self.async_client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            n = kwargs.get("n", 1),
            temperature= self.temperature,
            max_tokens=self.max_tokens,
            timeout=kwargs.get("timeout", 60)
        )
        return response.choices[0].message.content
    
    async def deal_tasks(self,tasks, max_concurrent_tasks=10):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        results = []

        async def sem_task(task):
            async with semaphore:
                return await task  # 注意这里是调用task()

        # 创建未执行的协程列表
        sem_tasks = [sem_task(task) for task in tasks]

        # 使用tqdm_asyncio.gather来调度任务并显示进度
        for coro in tqdm_asyncio.as_completed(sem_tasks, total=len(sem_tasks)):
            result = await coro
            results.append(result)

        return results
    

    def process_messages(self,messages):
        new_messages = []
        if "system" in messages:
            new_messages.append({"role":"system","content":messages["system"]}) 
        if "image" in messages:
            image = messages["image"]
            new_messages.append(
                {"role":"user","content":[
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image)}"}},
                    {"type":"text","text":messages["prompt"]}]
                    })
        elif "images" in messages:
            content = []
            for i,image in enumerate(messages["images"]):
                content.append({"type":"text","text":f"<image_{i+1}>: "})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image)}"}})
            content.append({"type":"text","text":messages["prompt"]})
            new_messages.append({"role":"user","content":content})
        else:
            new_messages.append({"role":"user","content":[{"type":"text","text":messages["prompt"]}]})
        
        messages = new_messages
        return messages


    def generate_output(self,messages):
        messages = self.process_messages(messages)
        try:
            response = self.response(messages)
        except Exception as e:
            # model = kwargs.get("model", self.model)
            print(f"get response failed: {e}")
            response = ""
        return response
    
    async def generate_output_async(self,messages,idx):
        messages = self.process_messages(messages)
        try:
            response = await self.response_async(messages)
            if not isinstance(response,str):
                response = ""
        except Exception as e:
            # model = kwargs.get("model", self.model)
            print(f"response failed: {e}")
            response = ""
        return response,idx
    
    def generate_outputs(self,messages_list):
        tasks = []
        for idx,messages in enumerate(messages_list):
            tasks.append(self.generate_output_async(messages,idx))
        results = asyncio.run(self.deal_tasks(tasks))
        results = sorted(results,key = lambda x:x[1])
        results = [result[0] for result in results]
        return results

