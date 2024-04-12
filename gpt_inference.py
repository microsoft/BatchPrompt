import os
import tiktoken

from gpt_model import GPT_Model
class LabelGen:
    def __init__(self, source, engine, dataset, batch_size):
        self.max_token = 1000
        self.batch_size = batch_size
        self.dataset = dataset
        
        system_prompt = self.read_file(f"./{self.dataset}/system_prompt.txt").replace("[BATCH_SIZE]", str(batch_size))
        example, label, few_shot_examples = self.few_shot_examples_gen()
        self.gpt_classifier = GPT_Model(source=source, engine=engine, system_prompt=system_prompt, input_prompt_prefix="", few_shot_examples=few_shot_examples)
        self.prompt_length = self.count_token(system_prompt) + self.count_token(example) + self.count_token(label)

    def read_file(self, filepath):
        with open(filepath, "r") as f:
            return f.read()
            
    def few_shot_examples_gen(self):
        few_shot_examples = []
        few_shot = self.read_file(f"./{self.dataset}/few_shot.txt")
        example, label = few_shot.split("====Answer====")
        few_shot_examples.append({"role":"user","content": example})
        few_shot_examples.append({"role":"assistant","content": label})
        return example, label, few_shot_examples
    
    def count_token(self, text):
        return(len(tiktoken.encoding_for_model('gpt-4').encode(text)))
        
    def total_prompt_length(self, input_text):
        return self.prompt_length + self.count_token(input_text)
        
    def gpt_label_gen(self, input_text, print_tag=False):
        result = self.gpt_classifier.generate(input_text=input_text, max_token=self.max_token)
        labels = self.label_gen(result)
        return labels
     
    def label_gen(self, gpt_res):
        labels = []
        gpt_res = gpt_res.split("[")
        for res in gpt_res[1:]:
            if 'class 2' in res or 'label 2' in res:
                labels.append(2)
            elif 'class 1' in res or 'label 1' in res:
                labels.append(1)
            elif 'class 0' in res or 'label 0' in res:
                labels.append(0)
        return labels

