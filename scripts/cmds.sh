#!/bin/sh

## split and save the model 
python split_and_save.py --model_name 'mistralai/Mistral-7B-Instruct-v0.1' --save_path "../2.16.0/model_store/mistralai/Mistral-7B-Instruct-v0.1/mistralai/Mistral-7B-Instruct-v0.1-split"
python compile.py compile