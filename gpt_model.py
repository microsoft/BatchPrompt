import time
import openai
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_fixed
)  # for exponential backoff


class GPT_Model:
    def __init__(self, source, engine, system_prompt, input_prompt_prefix, few_shot_examples=[]):
        openai.api_type = "azure"
        openai.api_version = "xxx"
        openai.api_base = "xxx"
        openai.api_key = "xxx"

        
        self.engine = engine
        self.system_prompt = system_prompt
        self.input_prompt_prefix = input_prompt_prefix
        self.few_shot_examples = few_shot_examples
        
    def _call_chatgpt(self, input_text, max_token, cnt=1):
        """
        Call chatGPT API for extraction of input text.

        Args:
            input_text (int): The first number.
            b (int): The second number.

        Returns:
            dict of list of str: A dictionary of lists of extracted events for each sentence
        """
        try:
            prompt_examples_msg = [{"role":"system","content":self.system_prompt}]
            prompt_examples_msg += self.few_shot_examples
            prompt_examples_msg += [{"role":"user","content":input_text}]
            
            return openai.ChatCompletion.create(
                        engine=self.engine,             
                        messages = prompt_examples_msg,
                        temperature=0,
                        max_tokens=max_token,
                        top_p=0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None)['choices'][0]['message']['content']
        except Exception as e:
            print(f"Exception: {e}")
            if "The response was filtered" in str(e):
                return ""
            raise Exception

    @retry(wait=wait_random_exponential(min=10, max=300), stop=stop_after_attempt(6))
    def generate(self, **kwargs):  # completion_with_backoff
        return self._call_chatgpt(**kwargs)