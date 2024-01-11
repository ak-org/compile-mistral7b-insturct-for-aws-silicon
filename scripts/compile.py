from transformers_neuronx.mistral.model import MistralForSampling
from transformers import AutoTokenizer
from transformers_neuronx import constants
from transformers_neuronx.config import NeuronConfig
import torch
import time
import os 
import sys 

# we will pin cores to 2 for inf2.xlarge 

os.environ['NEURON_RT_NUM_CORES'] = '2'
#os.environ["NEURON_CC_FLAGS"] = "-O3"  ## for best perf
version = '2.16.0'
model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
model_dir = f"../{version}/model_store/{model_id}/{model_id}-split"
model_compiled_dir = f"../{version}/model_store/{model_id}/neuronx_artifacts"
amp = 'bf16'
n_positions = 256

if sys.argv[1] == "compile":
    print('Current compilation is bugging. Exiting...')
    exit(-1)
    start = time.time()
    # Set sharding strategy for GQA to be shard over heads
    neuron_config = NeuronConfig(
        grouped_query_attention=constants.GQA.SHARD_OVER_HEADS
    )
    model = MistralForSampling.from_pretrained(
            model_dir,
            batch_size=1,
            n_positions=n_positions,
            tp_degree=int(os.environ['NEURON_RT_NUM_CORES']),
            amp=amp,
            neuron_config=neuron_config
            )
    model.to_neuron()
    # save model to the disk
    print(f'Saving model to {model_compiled_dir}')
    model.save(model_compiled_dir)
    elapsed = time.time() - start
    print(f'\nCompilation and loading took {elapsed} seconds\n')
elif sys.argv[1] == "infer":
    print('\n Loading pre-compiled model\n')
    ## load model from the disk
    start = time.time()
    # Set sharding strategy for GQA to be shard over heads
    neuron_config = NeuronConfig(
        grouped_query_attention=constants.GQA.SHARD_OVER_HEADS
    )
    model = MistralForSampling.from_pretrained(
            model_dir,
            batch_size=1,
            n_positions=n_positions,
            tp_degree=int(os.environ['NEURON_RT_NUM_CORES']),
            amp=amp,
            neuron_config=neuron_config
            )
    print(f'Loading model from {model_compiled_dir}')
    model.load(model_compiled_dir)
    model.to_neuron()
    elapsed = time.time() - start
    print(f'\n Model successfully loaded in {elapsed} seconds')
    # construct a tokenizer and encode prompt text
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    prompt = "[INST] Explain what a Mixture of Experts is in less than 100 words.[/INST]"
    input_ids = tokenizer(prompt, return_tensors='pt')

    # run inference with top-k sampling
    with torch.inference_mode():
        start = time.time()
        generated_sequences = model.sample(input_ids.input_ids, 
                                           sequence_length=256,
                                           top_k=50, 
                                           temperature=0.9,
                                           start_ids=None)
        elapsed = time.time() - start

    generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
    print(f'\ngenerated sequences {generated_sequences} in {elapsed} seconds\n')
else:
    print(f'\n**Missing paramter: Specify compiler or infer**\n')