#!/bin/bash
# please run them one by one intead of all of them to record the results.
python run.py --source='aiarch' --engine='aiarch-chatgpt' --dataset='rte' --batch_size=1 --num_vote=1 --early_stop=320 --json_save=True --load_json=True

python run.py --source='aiarch' --engine='aiarch-chatgpt' --dataset='rte' --batch_size=16 --num_vote=9 --early_stop=20 --json_save=True --load_json=True

python run.py --source='aiarch' --engine='aiarch-chatgpt' --dataset='rte' --batch_size=32 --num_vote=9 --early_stop=10 --json_save=True --load_json=True



python run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='rte' --batch_size=1 --num_vote=1 --early_stop=320 --json_save=True --load_json=True

python run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='rte' --batch_size=16 --num_vote=9 --early_stop=20 --json_save=True --load_json=True

python run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='rte' --batch_size=32 --num_vote=9 --early_stop=10 --json_save=True --load_json=True

python run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='rte' --batch_size=64 --num_vote=9 --early_stop=5 --json_save=True --load_json=True