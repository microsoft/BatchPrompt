from datasets import load_dataset
from tqdm import tqdm
from gpt_inference import LabelGen
import random
import argparse
import collections
import json
import math
import os
import matplotlib.pyplot as plt

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
    def __init__(self, source, engine, dataset, data_amount):
        DataLoader.__init__(self, dataset)
        self.data = self.load_data()
        
        self.data_amount = data_amount
        self.GPTLabel = LabelGen(source, engine, dataset, batch_size=1)
        self.prompt_template = self.read_file(f"./{self.dataset}/prompt_template.txt")

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
        with open(f'./{dir_path}/indices{self.data_amount}.json', 'w') as f:
            json.dump(indices, f)
        return indices
        

class BatchGPT(DataLoader):
    def __init__(self, source, engine, dataset, batch_size, num_vote, early_stop, move_type, mv_index=0):
        DataLoader.__init__(self, dataset)
        self.dataset = dataset
        self.num_vote = num_vote
        self.batch_size = batch_size
        self.engine = engine
        self.early_stop = early_stop
        self.move_type = move_type
        
        raw_data = self.load_data()
        self.raw_data_len = len(raw_data)
        data = self.load_filtered_data(raw_data)
        
        self.data_len = len(data)
        # [{'question':[q1,q2...], 'idx':[0,1...], 'label':[1,2...]}, ... {}]
        self.batched_data = [data[i:i+batch_size] for i in range(0, self.data_len, batch_size)]
        if mv_index > 0:
            self.mv_index = mv_index
            self.batched_data = self.mv_index_data()

        self.data_stat_init()
        self.prompt_template = self.read_file(f"./{self.dataset}/prompt_template.txt")
        self.GPTLabel = LabelGen(source, engine, dataset, batch_size)
        self.instruction_length = self.GPTLabel.prompt_length

    def load_filtered_data(self, raw_data):
        with open(f'./{self.dataset}/filtered_data/indices{self.early_stop*self.batch_size}.json', 'r') as f:
            indices = json.load(f)
        data = raw_data.filter(lambda x: x["idx"] in indices)
        return data

    def read_file(self, filepath):
        with open(filepath, "r") as f:
            return f.read() 
        
    def mv_index_data(self):
        rotate_batched_data = []
        for data_batch in self.batched_data:
            for key in data_batch:
                # permutate here for each key
                if self.move_type == 'rotate':
                    data_batch[key] = data_batch[key][-self.mv_index:] + data_batch[key][:-self.mv_index]
                elif self.move_type == 'insert':
                    data_batch[key].insert(self.mv_index, data_batch[key].pop(0))
            rotate_batched_data.append(data_batch)
        return rotate_batched_data

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
            #for dp_idx, dp_label in zip(data_batch['idx'], data_batch['label']):
            for batch_pos, (dp_idx, dp_label) in enumerate(zip(data_batch['idx'], data_batch['label'])):
                self.data_stat[dp_idx]['gt_label'] = dp_label
                self.data_stat[dp_idx]['gpt_voting'] = []
                self.data_stat[dp_idx]['batch_pos'] = batch_pos  # for each data point
                self.data_stat[dp_idx]['batch_idx'] = []         # for each voting
                
                
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
                if v > 0: data_batch = self.shuffle_batch(data_batch)  # shuffle data
                batch_prompt = self.prompt_gen(data_batch)
                gpt_labels =  self.GPTLabel.gpt_label_gen(batch_prompt)

                # update self.data_stat
                for data_idx, gpt_label in zip(data_batch['idx'], gpt_labels):
                    self.data_stat[data_idx]['gpt_voting'].append(gpt_label)
                    # record batch_idx for each voting
                    self.data_stat[data_idx]['batch_idx'].append(idx)
        if json_save:
            with open(f'./{self.dataset}/results/{self.engine}_bs{self.batch_size}_v{self.num_vote}_es{self.early_stop}.json', 'w') as f:
                json.dump(self.data_stat, f, indent=4)
        
    def majority_voting(self, data_stat, voting_times):
        # voting_times <= num_vote
        num_correct = total_num = 0
        # save results for each batch index
        pos_result = [[] for _ in range(self.batch_size)]
        for res in data_stat:
            if not res or not res['gpt_voting']: continue
            counter = collections.Counter(res['gpt_voting'][:voting_times])
            majority_element = counter.most_common(1)[0][0]
            num_correct += res['gt_label'] == majority_element
            total_num += 1
            
            # update results for each position
            pos_result[res['batch_pos']].append(res['gt_label'] == majority_element)
        return num_correct/total_num, pos_result
    

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
            with open(f'./{self.dataset}/results/{self.engine}_bs{self.batch_size}_v{self.num_vote}_es{self.early_stop}.json', 'r') as f:
                data_stat = json.load(f)
        else: data_stat = self.data_stat
        
        data_size = self.early_stop*self.batch_size if self.early_stop>0 else self.data_len
        num_batch = self.early_stop if self.early_stop>0 else self.data_len//self.batch_size
        for voting_times in range(1, self.num_vote+1, 2):
            acc, _ = self.majority_voting(data_stat, voting_times)
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

    def post_process_pos(self, load_json=True):
        if load_json:
            with open(f'./{self.dataset}/results/{self.engine}_bs{self.batch_size}_v{self.num_vote}_es{self.early_stop}.json', 'r') as f:
                data_stat = json.load(f)
        else: data_stat = self.data_stat

        _, pos_result = self.majority_voting(data_stat, 1)
        return pos_result
    
def parse_args():
    args = argparse.ArgumentParser(description="Get Accuracy and Efficiency Below: ")
    args.add_argument('--source', type=str, choices=['aiarch', 'msr'], default='aiarch') 
    args.add_argument('--engine', type=str, choices=['gpt-4-32k', 'aiarch-gpt-4-32k', 'aiarch-chatgpt'], default='aiarch-gpt-4-32k')
    args.add_argument('--dataset', type=str, choices=['rte', 'boolq', 'qqp'], default = 'boolq')
    args.add_argument('--batch_size', type=int, default=16) 
    args.add_argument('--num_vote', type=int, default=9)    
    args.add_argument('--early_stop', type=int, default=-1, help=" -1 means full dataset, >0 means only runs these batches")
    args.add_argument('--json_save', action="store_true", help=" if results are saved in json, you can do post_process next time without batch_process")
    args.add_argument('--load_json', action="store_true", help=" if results (json) must be loaded if batch_process is missing")
    args.add_argument('--move_type', type=str, choices=['insert', 'rotate'], default='rotate')  
    args = args.parse_args()
    return args

def draw_pos_fig(data, output_path):
    plt.plot(data, marker='*', markersize=10, linestyle='-')
    plt.xlabel("Batch Index")
    plt.ylabel("Accuracy")
    plt.savefig(output_path)

if __name__ == '__main__':
    """For Regular BPE Strategy
    args = parse_args()
    # prepare data (filter out some data points)
    if not os.path.isfile(f'./{args.dataset}/filtered_data/indices{args.early_stop*args.batch_size}.json'):
        DP =  DataPrepare(args.source, args.engine, args.dataset, args.early_stop*args.batch_size)
        DP.filter_data()
    
    # init
    BG = BatchGPT(args.source, args.engine, args.dataset, args.batch_size, args.num_vote, args.early_stop, args.move_type)
    
    # cannot skip this step when you firstly run this configuration
    # you can skip this step ONLY when the corresponding json saved in results folder
    BG.batch_process(json_save=args.json_save)
    
    # get acc and efficiency numbers
    BG.post_process(load_json=args.load_json)
    """

    """
    FOR CHECK LOGIC: try a small amount of data
    $ python run.py --source='aiarch' --engine='aiarch-chatgpt' --dataset='boolq' --batch_size=8 --num_vote=1 --early_stop=4 --json_save --load_json

    FOR EVAL: try 320 data, call 16 times with batch size of 16 (rotate)
    $ python run.py --source='aiarch' --engine='aiarch-chatgpt' --dataset='boolq' --batch_size=16 --num_vote=1 --early_stop=20 --json_save --load_json
    $ python run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='boolq' --batch_size=16 --num_vote=1 --early_stop=20 --json_save --load_json
    """
    args = parse_args()
    if not os.path.isfile(f'./{args.dataset}/filtered_data/indices{args.early_stop*args.batch_size}.json'):
        DP =  DataPrepare(args.source, args.engine, args.dataset, args.early_stop*args.batch_size)
        DP.filter_data()

    all_pos_res = [[] for _ in range(args.batch_size)]
    for i in range(args.batch_size):
        BG = BatchGPT(args.source, args.engine, args.dataset, args.batch_size, args.num_vote, args.early_stop, args.move_type, mv_index=i)
        BG.batch_process(json_save=args.json_save)
        pos_res = BG.post_process_pos(load_json=args.load_json)
        #print(pos_res)
        all_pos_res = [p + a for p, a in zip(pos_res, all_pos_res)]
    
    # final acc for pos
    all_pos_acc = [float(sum(a)/len(a)) for a in all_pos_res]
    print(all_pos_acc)
    out_path = f'./{args.dataset}/results/{args.engine}_bs{args.batch_size}_es{args.early_stop}.png'
    draw_pos_fig(all_pos_acc, out_path)