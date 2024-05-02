
# BatchPrompt

This repo saves the code for BatchPrompt Project.
Paper link:https://arxiv.org/abs/2309.00384#.

## Motivation
1. **Main Idea**: when we do data processing using LLM, we can do it in *Batch-fashion* to improve efficiency in terms of:

  ```The number of Calling```: = data size / batch size * number of voting

  ```The number of Token```: = system prompts + few shot prompts + input prompts

2. **Trade-off**: the bigger batch size, the higher efficiency but lower performance.
3. **Observation**: we find the accuracy oscillates with different locations within a batch.
4. **Hypothesis**: for this observation, we have several guesses/hypothesis: **(1)** data correlation for adjacent data points. **(2)** data bias due to different difficulty of data points.
5. **Solution**: permutation + voting. Permutation enables different data appear in different location to reduce the data correlation and bias due to different difficulty of data points. Voting helps to boost the Accuracy (self-consistency).
6. **Motivation**: With helps of **permutation + voting**, we improve accuracy at the price of efficiency. With the proposed **SEAS**, the lost efficiency was recovered. Compared to SinglePrompt (batch size=1), we improve efficiency by a lot without hurting accuracy. 


## Experiments
- **Step0**: understand your dataset by reviewing the **leaderboard** and checking some datapoints in `datasets_glue_superglue.ipynb`.

- **Step1**:
  - write system prompts in `./<your_dataset>/system_prompt.txt`.
  - write 3-5 few shot examples in `./<your_dataset>/few_shot.txt`. You don't have to change the code to print out demonstrations since you can find many demonstrations in `datasets_glue_superglue.ipynb`. Please **keep** the words *====Answer====* in *few_shot.txt*
  - write your template for **single** data point in `./<your_dataset>/prompt_template.txt`; Keep the word of **[INPUT_INDEX]** but you can change the word of *Input* if you want. All characters in **[]** must be **upper case** and they should exactly match the key name in your dataset. Key names for your dataset can be found in `datasets_glue_superglue.ipynb`.
  - Notice all the data format in these prompt files should match!

**Please notice step 2-4 are used for you to fine-tune the parameters by yourself on your own data or the provided data. You can ignore step 2-4 if you do not want to fine-tune the parameters. For model fine-tuning, SEAS is not included. You can directly conduct step5 to get the final results in the paper.**

- **Step2**: run a small amount of data, to figure out the best parameters you may need.
```
$ cd <folder_path>/BatchPrompt
$ sh <data_set>/run_small.sh
```
- **Step3**: repeat `Step2` to match the leaderboard accuracy by adjusting all prompts in `Step1`. If you find the accuracy is **too low**, you might need to adjust the model parameters, as can be found in the bash file.

- **Step4**: After getting the best parameter, keep all prompts and do not change them. Run 320 data for different configurations. Record all the results in `./<your_dataset>/results/result.md`. **Please note that in this step, "SEAS" is not included.**
$ cd <folder_path>/BatchPrompt
$ sh <data_set>/run_full.sh

- **Step5**: Run the final model, "BPE+SEAS". Record all the results in `./<your_dataset>/results/`. The original results in the paper have been saved in the results already.
$ cd <folder_path>/BatchPrompt
$ sh scripts/run_batch.sh
  
  **Tips 1**: Please notice we have a automatic sub-step in Step4 to filter out the sensitive data. in Step4, you can firstly run **chatgpt** experiments to get **indices320.json**, which are data filtered by sensitive information. We assume that chatgpt and gpt4 use the same filtering so that run chatgpt first will skip this filtering in gpt4 experiments. After you get this filtered data, you can all experiments in parallel.
  
 **Tips 2**:
 ```
$ run.py
```
is used for model parameter fine-tuning, without SEAS.
 ```
$ adaptive_batch_run.py
```
is the model we proposed in the paper, including **BatchPrompt+BPE+SEAS**.

  
