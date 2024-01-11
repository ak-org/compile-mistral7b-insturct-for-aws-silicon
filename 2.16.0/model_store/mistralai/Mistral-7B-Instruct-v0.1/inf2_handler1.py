import logging
import os
from abc import ABC
from threading import Thread
import json

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers_neuronx.mistral.model import MistralForSampling
from transformers_neuronx import constants
from ts.handler_utils.micro_batching import MicroBatching
from ts.torch_handler.base_handler import BaseHandler
from transformers_neuronx.config import NeuronConfig

logger = logging.getLogger(__name__)
os.environ['NEURON_RT_NUM_CORES'] = '2'
os.environ["NEURON_CC_FLAGS"] = "-O3"  ## for best perf

class LLMHandler(BaseHandler, ABC):
    """
    Transformers handler class for text completion streaming on Inferentia2
    """
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.n_positions = None
        self.tokenizer = None
        self.output_streamer = None
        # enable micro batching
        self.handle = MicroBatching(self)
        
    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        model_checkpoint_dir = ctx.model_yaml_config.get("handler", {}).get(
            "model_checkpoint_dir", ""
        )
        model_compiled_dir = ctx.model_yaml_config.get("handler", {}).get(
            "model_compiled_weights", ""
        )
        model_checkpoint_path = f"{model_dir}/{model_checkpoint_dir}"
        model_compiled_path = f"{model_dir}/{model_compiled_dir}"

        # micro batching initialization
        micro_batching_parallelism = ctx.model_yaml_config.get(
            "micro_batching", {}
        ).get("parallelism", None)
        if micro_batching_parallelism:
            logger.info(
                f"Setting micro batching parallelism  from model_config_yaml: {micro_batching_parallelism}"
            )
            self.handle.parallelism = micro_batching_parallelism

        micro_batch_size = ctx.model_yaml_config.get("micro_batching", {}).get(
            "micro_batch_size", 1
        )
        logger.info(f"Setting micro batching size: {micro_batch_size}")
        self.handle.micro_batch_size = micro_batch_size

        # settings for model compiliation and loading
        amp = ctx.model_yaml_config.get("handler", {}).get("amp", "bf16")
        tp_degree = ctx.model_yaml_config.get("handler", {}).get("tp_degree", 2)
        self.n_positions = ctx.model_yaml_config.get("handler", {}).get("n_positions", 256)
        self.top_k = ctx.model_yaml_config.get("handler", {}).get("top_k", 50)
        self.top_p = ctx.model_yaml_config.get("handler", {}).get("top_p", 0.9)
        self.temperature = ctx.model_yaml_config.get("handler", {}).get("temperature", 0.5)

        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path)
        self.tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
        logger.info("Starting to load the model")
        neuron_config = NeuronConfig(
            grouped_query_attention=constants.GQA.SHARD_OVER_HEADS
        )
        self.model = MistralForSampling.from_pretrained(
            model_checkpoint_path,
            batch_size=self.handle.micro_batch_size,
            n_positions=self.n_positions,
            amp=amp,
            tp_degree=tp_degree,
        )
        self.model.load(model_compiled_path)
        self.model.to_neuron()
        logger.info("Model loaded in the Neuron cores.")
        model_config = AutoConfig.from_pretrained(model_checkpoint_path)
        
        # Use the `HuggingFaceGenerationModelAdapter` to access the generate API
        # self.model = HuggingFaceGenerationModelAdapter(model_config, self.model)
        ## the following was added in response to following error
        ## ValueError: If `eos_token_id` is defined, make sure that `pad_token_id` is defined.
        # self.model.config.pad_token_id = self.model.config.eos_token_id
        self.initialized = True
        
    def preprocess(self, requests):
        ######################## 
        ## 
        ## request must confirm to following template:
        ## {
        ##        "inputs": "prompt goes here" # mandatory
        ##        "parameters": { #optional
        ##            'n_positions': 50, 
        ##            'top_p': 0.9, 
        ##            'top_k': 50, 
        ##            'temperature': 0.6
        ##       }
        ## }
        input_text = []
        input_payload = []
        for req in requests:
            data = req.get("data") or req.get("body")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            logger.info(f"received req={data}")
            input_payload.append(data.strip())
        logger.info(f"Input request for Inference is {input_payload}")
        inference_payload = json.loads(input_payload[0])
        logger.info(f"Input json for Inference is {inference_payload}")
        input_text.append(inference_payload['inputs'])
        self.input_lenth = len(input_text)
        prompt = input_text[0]
        logger.info(f"Input text for Inference is {prompt}")
        logger.info(f"Input text Length is {self.input_lenth}")
        ## if paramters are specified in JSON payload,
        ## override the default model config paramters
        if 'parameters' in inference_payload:
            if 'temperature' in inference_payload['parameters']:
                self.temperature = inference_payload['parameters']['temperature']
                self.do_sample = True
            if 'top_k' in inference_payload['parameters']:
                self.top_k = inference_payload['parameters']['top_k']
            if 'top_p' in inference_payload['parameters']:
                self.top_p = inference_payload['parameters']['top_p']
            if 'n_positions' in inference_payload['parameters']:
                self.n_positions = inference_payload['parameters']['n_positions']
        # Ensure the compiled model can handle the input received
        if self.input_lenth > self.handle.micro_batch_size:
            raise ValueError(
                f"Model is compiled for batch size {self.handle.micro_batch_size} but received input of size {self.input_lenth}"
            )
        # Pad input to match compiled model batch size
        # will be zero
        input_text.extend([""] * (self.handle.micro_batch_size - self.input_lenth))
        
        logger.info(f"Prompt after padding is {prompt}")
        logger.info(f"Input text Length after padding is {len(input_text)}")
        encoded_inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        return encoded_inputs

    
    def inference(self, encoded_inputs):
        logging.info(f"Input paramters to sample function {self.top_p}, {self.top_k}, {self.temperature}, {self.n_positions}")
        with torch.inference_mode():
            generated_sequences = self.model.sample(encoded_inputs, 
                                            sequence_length=self.n_positions, 
                                            top_k=self.top_k,
                                            top_p=self.top_p,
                                            temperature=self.temperature)
        generated_sequences = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_sequences]
        logger.info(f'Generated Inference is {generated_sequences}')
        return generated_sequences
    
    def postprocess(self, inference_output):
        logger.info('Returning ')
        logger.info(json.dumps({
            "generated_text": inference_output[0]
        }))
        return [
                json.dumps({"generated_text": inference_output[0]})
            ]