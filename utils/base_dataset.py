import torch
import os
import json

from tqdm import tqdm
import gc

from .utils import save_json

class BaseDataset:
  def __init__(self):
    self.chunk_idx = int(os.environ.get("chunk_idx",0))
    self.num_chunks = int(os.environ.get("num_chunks",1))

  def run(self,samples,model,batch_size = 2000):
    out_samples = []
    with torch.no_grad():
        messages_list = []
        current_messages = []
        current_samples = []
        for sample in tqdm(samples):
            messages = sample["messages"]
            current_messages.append(messages)
            current_samples.append(sample)
            if len(current_messages) >= batch_size:
                messages_list.append([current_messages,current_samples])
                current_messages = []
                current_samples = []
        if current_messages:
            messages_list.append([current_messages,current_samples])
        
        for current_messages,current_samples in tqdm(messages_list):
            outputs = model.generate_outputs(current_messages)
            try:
                for sample,response in zip(current_samples,outputs):
                    del sample["messages"]
                    sample["response"] = response
                    out_samples.append(sample)   
            except Exception as e:
                from pdb import set_trace;set_trace()
                print(e)
            gc.collect()
    return out_samples

  def cal_matrics(self):
    pass

  def init_dataset(self):
    pass

  def construct_messages(self):
    pass

  def eval(self):
      model = self.model
      dataset_path = self.dataset_path
      output_path = self.output_path
      num_chunks = self.num_chunks
      chunk_idx = self.chunk_idx
      if num_chunks == 1:
          results_path = os.path.join(output_path,"results.json")
          matric_path = os.path.join(output_path,"metrics.json")
          out_samples = self.run(self.samples,model)
          save_json(results_path,out_samples)

          metrics,out_samples = self.cal_metrics(out_samples)
          save_json(matric_path,metrics)
          save_json(results_path,out_samples)
          return metrics

      elif num_chunks > 1:
        results_path = os.path.join(output_path,f"results_{chunk_idx}.json")
        final_results_path = os.path.join(output_path,"results.json")
        out_samples = self.run(self.samples,model)
        save_json(results_path,out_samples)

        total_results_path = os.listdir(output_path)
        total_results_path = [result for result in total_results_path if result.startswith("results_")]
        if len(total_results_path) == num_chunks:
            total_results = []
            for result in total_results_path:
                results_path = os.path.join(output_path,result)
                with open(results_path,"r") as f:
                    total_results.extend(json.load(f))

            save_json(final_results_path,total_results)
            metrics,out_samples = self.cal_metrics(total_results)
            matric_path = os.path.join(output_path,"metrics.json")
            save_json(matric_path,metrics)
            save_json(final_results_path,out_samples)
            return metrics
        else:
            return None
      else:
          raise ValueError("num_chunks must be greater than 0")
  
  def _download_file_local(self,local_path,url):
        # download the specific file to local_path
        
        os.makedirs(local_path,exist_ok=True)
        
        # Extract filename from URL
        filename = url.split("/")[-1]
        file_path = os.path.join(local_path, filename)
        
        # Check if wget or curl is available
        if os.system("which wget > /dev/null 2>&1") == 0:
            download_cmd = f"wget {url} -O {file_path}"
        elif os.system("which curl > /dev/null 2>&1") == 0:
            download_cmd = f"curl -L {url} -o {file_path}"
        else:
            raise RuntimeError("Neither wget nor curl is available for downloading")

        # Download with error handling
        if os.system(download_cmd) != 0:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise RuntimeError("Failed to download dataset")
        
  def _unzip_img_zip_local(self, local_path, zip_filename):
        # suppose zip_filename is like 'images.zip' or 'data.tgz'
        zip_file_path = os.path.join(local_path, zip_filename)
        
        if zip_filename.endswith('.zip'):
            if os.system(f"unzip -q {zip_file_path} -d {local_path}") != 0:
                if os.path.exists(zip_file_path):
                    os.remove(zip_file_path)
                raise RuntimeError("Failed to extract dataset")
        elif zip_filename.endswith('.tgz') or zip_filename.endswith('.tar.gz'):
            if os.system(f"tar -xzf {zip_file_path} -C {local_path}") != 0:
                if os.path.exists(zip_file_path):
                    os.remove(zip_file_path)
                raise RuntimeError("Failed to extract dataset")
        else:
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
            raise RuntimeError(f"Unsupported file format: {zip_filename}")
        
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)