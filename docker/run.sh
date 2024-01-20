docker run -v ./2.16.1:/2.16.1 --device /dev/neuron0:/dev/neuron0 --rm -it --entrypoint /bin/bash 102048127330.dkr.ecr.us-east-1.amazonaws.com/neuronx-torch2:2.16.1

## torchserve --start --ncs --model-store /2.16.0/model_store --models llama-2-7b-chat