from transformers_neuronx.mistral.model import MistralForSampling
from transformers import AutoTokenizer
from transformers_neuronx import constants
from transformers_neuronx.config import NeuronConfig
import torch
import time
import os 
import sys 

tp_degree = int(sys.argv[2])
#os.environ["NEURON_CC_FLAGS"] = "-O3"  ## for best perf
version = '2.16.1'
model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
model_dir = f"../{version}/model_store/{model_id}/Mistral-7B-Instruct-v0.1-split"
model_compiled_dir = f"../{version}/model_store/{model_id}/neuronx_artifacts-{tp_degree}"
amp = 'bf16'
if tp_degree == 2:
    neuron_config = NeuronConfig(
        grouped_query_attention=constants.GQA.SHARD_OVER_HEADS
    )
    n_positions = 1024
else:
    neuron_config = NeuronConfig(
        grouped_query_attention=constants.GQA.REPLICATED_HEADS
    )
    n_positions = 4096 

if sys.argv[1] == "compile":
    start = time.time()
    # Set sharding strategy for GQA to be shard over heads    
    model = MistralForSampling.from_pretrained(
            model_dir,
            batch_size=1,
            n_positions=n_positions,
            tp_degree=tp_degree,
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
    model = MistralForSampling.from_pretrained(
            model_dir,
            batch_size=1,
            n_positions=n_positions,
            tp_degree=tp_degree,
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

    prompt2 = """[INST]What is your favorite condiment?[/INST]"""
    input_ids = tokenizer(prompt2, return_tensors='pt')
    # run inference with top-k sampling
    with torch.inference_mode():
        start = time.time()
        generated_sequences = model.sample(input_ids.input_ids, 
                                           sequence_length=n_positions,
                                           top_k=50, 
                                           temperature=0.9,
                                           start_ids=None)
        elapsed = time.time() - start
    generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
    print(f'\ngenerated sequences {generated_sequences} in {elapsed} seconds\n')
    
    prompt = """[INST]
    "instruction": "What is a dispersive prism?", 
    "context": "In optics, a dispersive prism is an optical prism that is used to disperse light, that is, to separate light into its spectral components (the colors of the rainbow). Different wavelengths (colors) of light will be deflected by the prism at different angles. This is a result of the prism material's index of refraction varying with wavelength (dispersion). Generally, longer wavelengths (red) undergo a smaller deviation than shorter wavelengths (blue). The dispersion of white light into colors by a prism led Sir Isaac Newton to conclude that white light consisted of a mixture of different colors.", 
    "response": [/INST]
    """
    input_ids = tokenizer(prompt, return_tensors='pt')
    # run inference with top-k sampling
    with torch.inference_mode():
        start = time.time()
        generated_sequences = model.sample(input_ids.input_ids, 
                                           sequence_length=n_positions,
                                           top_k=50, 
                                           temperature=0.9,
                                           start_ids=None)
        elapsed = time.time() - start
    generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
    print(f'\ngenerated sequences {generated_sequences} in {elapsed} seconds\n')
else:
    print(f'\n**Missing paramter: Specify compile or infer**\n')