# Context
This code repository shows how you can download a model from huggingface repository, compile it for `Neuronx` platform and deploy it as SageMaker Inference endpoint.

# Pre-requisite
You will need an EC2 instance of instance size **inf2.24xlarge** to run the code, specifically the model compilation code.

## Repository Structure
| Directory Name | Purpose |
|----------------|--------------------------------------------
| 2.16.1         | This directory will store model artifacts  
| docker         | Contains scripts to build & push docker image to ECR, run docker locally and it  also contains Dockerfile to facilitate local build of Docker image
| Scripts        |  This folder contains script to a. download model from huggingface b. split and save the model locally c. compile the model for Neuronx platform 
| smep           | This folder contains scripts to deploy `Neuronx` compiled model as SageMaker inference endpoint using torchserve based container. It also contains script to run inferences, get status of SageMaker endpoint and to delete the SageMaker inference endpoint.
| smep-with-lmi  | This folder contains scripts to deploy `Neuronx` compiled model as SageMaker inference endpoint using deep learning java (DLJ) based container. 

## Steps 

1. If want to use `torchserve` based container. Use `build_and_push.sh` script to build the container image, push it to your ECR repository and use `run.sh` to run the container locally.

2. Model compilation

    a. Use `cmds.sh` script to download model from huggingface and split & save the model.

    b. Use `compile.py` script to compile the model for Neuronx. Model artifacts will be saved in the 2.16.1 folder 

3. To Deploy model as SageMaker Inference endpoint, run following commands (either in smep or smep-with-lmi folder)

    a. `deploy.py` to deploy the inference endpoint 
    b  `infer.py` to run inference against deployed SageMaker Inference endpoint
    c. `cleanup.py` to delete the endpoint and cleanup 


 


