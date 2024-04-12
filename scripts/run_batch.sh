#!/bin/bash
python adaptive_batch_run.py --source='aiarch' --engine='aiarch-chatgpt' --dataset='boolq' --batch_size=32 --num_vote=9 --early_stop=10 --json_save=True --load_json=True --weight --weights=5 --confidence_type='single'

python adaptive_batch_run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='boolq' --batch_size=32 --num_vote=9 --early_stop=10 --json_save=True --load_json=True --weight --weights=5 --confidence_type='single'

python adaptive_batch_run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='boolq' --batch_size=64 --num_vote=9 --early_stop=5 --json_save=True --load_json=True --weight --weights=5 --confidence_type='single'

python adaptive_batch_run.py --source='aiarch' --engine='aiarch-chatgpt' --dataset='qqp' --batch_size=32 --num_vote=9 --early_stop=10 --json_save=True --load_json=True --weight --weights=5 --confidence_type='single'

python adaptive_batch_run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='qqp' --batch_size=32 --num_vote=9 --early_stop=10 --json_save=True --load_json=True --weight --weights=5 --confidence_type='single'

python adaptive_batch_run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='qqp' --batch_size=64 --num_vote=9 --early_stop=5 --json_save=True --load_json=True --weight --weights=5 --confidence_type='single'

python adaptive_batch_run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='qqp' --batch_size=160 --num_vote=9 --early_stop=2 --json_save=True --load_json=True --weight --weights=5 --confidence_type='single'


python adaptive_batch_run.py --source='aiarch' --engine='aiarch-chatgpt' --dataset='rte' --batch_size=32 --num_vote=9 --early_stop=10 --json_save=True --load_json=True --weight --weights=5 --confidence_type='single'

python adaptive_batch_run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='rte' --batch_size=32 --num_vote=9 --early_stop=10 --json_save=True --load_json=True --weight --weights=5 --confidence_type='single'

python adaptive_batch_run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='rte' --batch_size=64 --num_vote=9 --early_stop=5 --json_save=True --load_json=True --weight --weights=5 --confidence_type='single'

python adaptive_batch_run.py --source='aiarch' --engine='aiarch-gpt-4-32k' --dataset='rte' --batch_size=160 --num_vote=9 --early_stop=2 --json_save=True --load_json=True --weight --weights=5 --confidence_type='single'
