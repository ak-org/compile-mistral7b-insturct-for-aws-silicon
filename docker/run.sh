docker run -v ./2.16.0:/2.16.0 --device /dev/neuron0:/dev/neuron0 --rm -it --entrypoint /bin/bash 102048127330.dkr.ecr.us-east-1.amazonaws.com/neuronx:2.16.0

## torchserve --start --ncs --model-store /2.16.0/model_store --models llama-2-7b-chat