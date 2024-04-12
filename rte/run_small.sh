#!/bin/bash
python run.py --source='aiarch' --engine='aiarch-chatgpt' --dataset='rte' --batch_size=1 --num_vote=1 --early_stop=200 --json_save=True --load_json=True
