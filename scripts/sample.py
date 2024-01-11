import torch
from transformers_neuronx import constants
from transformers_neuronx.mistral.model import MistralForSampling
from transformers_neuronx.module import save_pretrained_split
from transformers_neuronx.config import NeuronConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys


# Load and save the CPU model with bfloat16 casting
model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
if len(sys.argv) > 1 and sys.argv[1] == "split":
    model_cpu = AutoModelForCausalLM.from_pretrained(model_id)
    save_pretrained_split(model_cpu, f'{model_id}-split')
else:
    # Set sharding strategy for GQA to be shard over heads
    neuron_config = NeuronConfig(
        grouped_query_attention=constants.GQA.REPLICATED_HEADS
        #grouped_query_attention=constants.GQA.SHARD_OVER_HEADS
    )

    # Create and compile the Neuron model
    model_neuron = MistralForSampling.from_pretrained(f'{model_id}-split', 
                                                    batch_size=1, 
                                                    tp_degree=2, 
                                                    n_positions=256, 
                                                    amp='bf16', 
                                                    neuron_config=neuron_config)
    model_neuron.to_neuron()

    # Get a tokenizer and example input
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text = "[INST] Explain what a Mixture of Experts is in less than 100 words. [/INST]"
    encoded_input = tokenizer(text, return_tensors='pt')

    # Run inference
    with torch.inference_mode():
        generated_sequence = model_neuron.sample(encoded_input.input_ids, 
                                                sequence_length=256,
                                                start_ids=None)
        print([tokenizer.decode(tok) for tok in generated_sequence])