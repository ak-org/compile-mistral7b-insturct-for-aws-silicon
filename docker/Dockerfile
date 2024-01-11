ARG BASE_IMAGE

FROM $BASE_IMAGE
RUN apt-get update && apt-get upgrade -y 
## pin the library versions for SDK 2.16.0
RUN pip install --upgrade pip
RUN pip install --upgrade \
                transformers-neuronx==0.9.474 \
                neuronx-cc==2.12.54.0 \
                torch-neuronx==2.0.0.2.0.1b0 \
                --extra-index-url=https://pip.repos.neuron.amazonaws.com
RUN pip install --upgrade torchserve==0.9.0 sentencepiece==0.1.99
ENV install_py_dep_per_model=true
