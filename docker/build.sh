#!/usr/bin/env bash
region="us-east-1"
baseimage="763104351884.dkr.ecr.$region.amazonaws.com/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.14.1-ubuntu20.04"
#baseimage="763104351884.dkr.ecr.$region.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.15.0-ubuntu20.04"
reponame="neuronx-torch2"
versiontag="2.16.0"
account="102048127330"

./build_and_push.sh $reponame $versiontag $baseimage $region $account