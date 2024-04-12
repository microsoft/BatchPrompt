from datasets import load_dataset
from tqdm import tqdm
from gpt_inference import LabelGen
import random
import argparse
import collections
import json
import math
import os

class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_info = {}
        self.data_info["boolq"] = {"collection": "super_glue", "test_set": "validation"}
        self.data_info["rte"] = {"collection": "super_glue", "test_set": "validation"}
        self.data_info["qqp"] = {"collection": "glue", "test_set": "validation"}
        
    def load_data(self):
        info1 = self.data_info[self.dataset]["collection"]
        info2 = self.data_info[self.dataset]["test_set"]
        data = load_dataset(info1, self.dataset)[info2]
        shuffled_data = data.shuffle()
        return shuffled_data

class DataPrepare(DataLoader):
    def __init__(self, source, engine, dataset, data_amount, weight, weights, confidence_type):
        DataLoader.__init__(self, dataset)
        self.data = self.load_data()
        self.weight = weight
        self.data_amount = data_amount
        batch_size=1
        self.GPTLabel = LabelGen(source, engine, dataset, batch_size, weight, weights, confidence_type)
        self.prompt_template = self.read_file(f"./{self.dataset}/prompt_template.txt")
        self.engine = engine

    def read_file(self, filepath):
        with open(filepath, "r") as f:
            return f.read() 
            
    def prompt_gen(self, data_point):
        data_prompt = self.prompt_template                            # init
        data_prompt = data_prompt.replace(f"[INPUT_INDEX]", str(0))   # batch=1
        for key in data_point.keys():
            if f"[{key.upper()}]" not in data_prompt: continue
            data_prompt = data_prompt.replace(f"[{key.upper()}]", data_point[key])
        return data_prompt

    def filter_data(self):
        indices = []
        for data_point in self.data:
            if len(indices) == self.data_amount: break
            
            data_prompt = self.prompt_gen(data_point)
            gpt_labels =  self.GPTLabel.gpt_label_gen(data_prompt)
            if gpt_labels:
                indices.append(data_point['idx'])
        
        dir_path = f'./{self.dataset}/filtered_data/'
        if not os.path.isdir(dir_path): os.makedirs(dir_path)
        with open(f'{dir_path}/indices{self.data_amount}.json', 'w') as f:
            json.dump(indices, f)
        return indices
        



class BatchGPT(DataLoader):
    def __init__(self, source, engine, dataset, batch_size, num_vote, early_stop, weight, weights, confidence_type):
        DataLoader.__init__(self, dataset)
        self.dataset = dataset
        self.num_vote = num_vote
        self.batch_size = batch_size
        self.engine = engine
        self.early_stop = early_stop
        self.weight = weight
        self.weights = weights
        self.confidence_type = confidence_type
        raw_data = self.load_data()
        self.raw_data_len = len(raw_data)
        data = self.load_filtered_data(raw_data)
        
        self.data_len = len(data)
        # [{'question':[q1,q2...], 'idx':[0,1...], 'label':[1,2...]}, ... {}]
        self.batched_data = [data[i:i+batch_size] for i in range(0, self.data_len, batch_size)]   
        self.data_stat_init()

        self.prompt_template = self.read_file(f"./{self.dataset}/prompt_template.txt")
        self.GPTLabel = LabelGen(source, engine, dataset, batch_size, weight, weights, confidence_type)
        self.instruction_length = self.GPTLabel.prompt_length


    def load_filtered_data(self, raw_data):
        with open(f'./{self.dataset}/filtered_data/indices{self.early_stop*self.batch_size}.json', 'r') as f:
            indices = json.load(f)
        data = raw_data.filter(lambda x: x["idx"] in indices)
        return data

    def read_file(self, filepath):
        with open(filepath, "r") as f:
            return f.read() 
    
    def prompt_gen(self, data_batch):
        batch_prompt = ""
        for index in range(len(data_batch['idx'])):
            data_prompt = self.prompt_template        # init
            data_prompt = data_prompt.replace(f"[INPUT_INDEX]", str(index))
            for key in data_batch.keys():
                if f"[{key.upper()}]" not in data_prompt: continue
                data_prompt = data_prompt.replace(f"[{key.upper()}]", data_batch[key][index])
            
            # update self.data_stat num_token
            dp_idx = data_batch['idx'][index]
            if 'num_token' not in self.data_stat[dp_idx]: 
                self.data_stat[dp_idx]['num_token'] = self.GPTLabel.count_token(data_prompt)
            
            batch_prompt += data_prompt + 2*"\n"
        return batch_prompt
    
    def data_stat_init(self):
        self.data_stat = [{} for _ in range(self.raw_data_len)]     # json: [{"gt_label": 0/1/2, "gpt_voting": [0, 0, 1 ...], "num_token":80}, {empty}, {}... ]
        for data_batch in self.batched_data:
            for dp_idx, dp_label in zip(data_batch['idx'], data_batch['label']):
                self.data_stat[dp_idx]['gt_label'] = dp_label
                self.data_stat[dp_idx]['gpt_voting'] = []
                self.data_stat[dp_idx]['batch_idx'] = []
                
                
    def shuffle_batch(self, data_batch):
        shuffled_data_batch = {}
        indices = list(range(len(data_batch['idx'])))
        random.shuffle(indices)

        for key, value in data_batch.items():
            shuffled_value = [value[i] for i in indices]
            shuffled_data_batch[key] = shuffled_value
        return shuffled_data_batch
    
    def batch_process(self, json_save=True):
        # early stop for small data testing
        for idx, data_batch in tqdm(enumerate(self.batched_data)):
            if self.early_stop > 0 and idx == self.early_stop: break
            for v in range(self.num_vote):
                if data_batch['label'] == []:
                    continue
                if v > 0: data_batch = self.shuffle_batch(data_batch)  # shuffle data
                batch_prompt = self.prompt_gen(data_batch)
                gpt_labels =  self.GPTLabel.gpt_label_gen(batch_prompt)
                idx_in_batch = 0
                delete_idx = []
                # update self.data_stat
                for data_idx, gpt_label in zip(data_batch['idx'], gpt_labels):
                    self.data_stat[data_idx]['gpt_voting'].append(gpt_label)                    
                    ### adaptively delete
### time 1 if len(self.data_stat[data_idx]['gpt_voting'])>1 and len(gpt_label) == 5 and len(self.data_stat[data_idx]['gpt_voting'][-2]) == 5 and self.data_stat[data_idx]['gpt_voting'][-2][0] == self.data_stat[data_idx]['gpt_voting'][-1][0]:
                    if len(self.data_stat[data_idx]['gpt_voting'])>1 and self.data_stat[data_idx]['gpt_voting'][-2][0] == self.data_stat[data_idx]['gpt_voting'][-1][0]:
                        delete_idx.append(idx_in_batch)
                    # record batch_idx
                    self.data_stat[data_idx]['batch_idx'].append(idx)
                    idx_in_batch += 1
                if delete_idx != []:   
                    deleted_num = 0
                    for item in delete_idx:
                        item = item - deleted_num
                        for key in data_batch.keys():
                            data_batch[key].remove(data_batch[key][item])
                        deleted_num += 1

                         
        if json_save:
            with open(f'./{self.dataset}/results/{self.weight}/{self.engine}_bs{self.batch_size}_v{self.num_vote}_es{self.early_stop}.json', 'w') as f:
                json.dump(self.data_stat, f, indent=4)

    def majority_voting(self, data_stat, voting_times):
        # voting_times <= num_vote
        num_correct = total_num = 0        
        for res in data_stat:
            if not res or not res['gpt_voting']: continue
            counter = collections.Counter([v for r in res['gpt_voting'][:voting_times] for v in r])
            majority_element = counter.most_common(1)[0][0]
            num_correct += res['gt_label'] == majority_element
            total_num += 1
        return num_correct/total_num
    

    def num_token(self, data_stat, voting_times):
        total_voting_token = total_baseline_token = 0
        for res in data_stat:
            if not res or not res['gpt_voting'] : continue            
            if 'num_token' in res: 
                total_voting_token += res['num_token'] * min(voting_times, len(res['gpt_voting']))
                total_baseline_token += self.instruction_length + res['num_token']
        return total_baseline_token, total_voting_token


    def post_process(self, load_json=True):
        if load_json:
            with open(f'./{self.dataset}/results/{self.weight}/{self.engine}_bs{self.batch_size}_v{self.num_vote}_es{self.early_stop}.json', 'r') as f:
                data_stat = json.load(f)
        else: data_stat = self.data_stat
        
        data_size = self.early_stop*self.batch_size if self.early_stop>0 else self.data_len
        num_batch = self.early_stop if self.early_stop>0 else self.data_len//self.batch_size
        for voting_times in range(1, self.num_vote+1, 2):
            acc = self.majority_voting(data_stat, voting_times)
            print(f"[{self.dataset.upper()}] ===== Voting {voting_times} Times =====")
            print(f"[{self.dataset.upper()}] Accuracy is {acc*100}%!")
            
            # number of calling
            num_call = math.ceil(data_size/self.batch_size) * voting_times
            save_percent = 1 - float(num_call/data_size)
            print(f"[{self.dataset.upper()}] Baseline Calling {data_size} times!")
            print(f"[{self.dataset.upper()}] BatchGPT Calling {num_call} times, save {save_percent*100}%!")
            
            # number of token
            total_baseline_token, total_voting_token = self.num_token(data_stat, voting_times)
            total_voting_token += num_batch * self.instruction_length
            save_percent = 1 - float(total_voting_token/total_baseline_token)
            print(f"[{self.dataset.upper()}] Baseline TokenSize {total_baseline_token}!")
            print(f"[{self.dataset.upper()}] BatchGPT TokenSize {total_voting_token}, save {save_percent*100}%!")
            print("\n")



def parse_args():
    args = argparse.ArgumentParser(description="Get Accuracy and Efficiency Below: ")
    args.add_argument('--source', type=str, choices=['aiarch', 'msr'], default='aiarch') 
    args.add_argument('--engine', type=str, choices=['gpt-4-32k', 'aiarch-gpt-4-32k', 'aiarch-chatgpt'], default='aiarch-chatgpt')
    args.add_argument('--dataset', type=str, choices=['rte', 'boolq', 'qqp'], default = 'boolq')
    args.add_argument('--batch_size', type=int, default=32) 
    args.add_argument('--num_vote', type=int, default=9)    
    args.add_argument('--early_stop', type=int, default=2, help=" -1 means full dataset, >0 means only runs these batches")
    args.add_argument('--json_save', type=bool, default=True, help=" if results are saved in json, you can do post_process next time without batch_process")
    args.add_argument('--load_json', type=bool, default=True, help=" if results (json) must be loaded if batch_process is missing") 
    args.add_argument('--weight', help=" if True confidence will be used", action="store_true") 
    args.add_argument('--weights', type=int, default=5) 
    args.add_argument('--confidence_type', type=str, choices=['binary', 'single'], default='binary')    
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # prepare data (filter out some data points)
    if not os.path.isfile(f'./{args.dataset}/filtered_data/indices{args.early_stop*args.batch_size}.json'):
        DP =  DataPrepare(args.source, args.engine, args.dataset, args.early_stop*args.batch_size, args.weight, args.weights, args.confidence_type)
        DP.filter_data()
    
    # init
    BG = BatchGPT(args.source, args.engine, args.dataset, args.batch_size, args.num_vote, args.early_stop, args.weight, args.weights, args.confidence_type)
    
    # cannot skip this step when you firstly run this configuration
    # you can skip this step ONLY when the corresponding json saved in results folder
    BG.batch_process(json_save=args.json_save)
    
    # get acc and efficiency numbers
    BG.post_process(load_json=args.load_json)